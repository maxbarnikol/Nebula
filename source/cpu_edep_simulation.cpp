#include "cpu_edep_simulation.h"

#include "common/work_pool.h"
#include "drivers/cpu/energydeposit_cpu_driver.h"
#include "geometry/octree.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace nbl::cpu_edep {
namespace {

using geometry_t = nbl::geometry::octree<false>;
using base_driver_t =
    nbl::drivers::energydeposit_cpu_driver<scatter_physics<false>, intersect_t,
                                          geometry_t>;

constexpr std::uint32_t kInvalidElectronTag = 0xFFFFFFFFu;

class streaming_driver : public base_driver_t {
public:
  using base_driver_t::base_driver_t;

  using particle_index_t = typename base_driver_t::particle_index_t;

  template <typename emit_fn>
  void simulate_to_end(const std::vector<int2> &pixels,
                       bool emit_secondary_spawn_events,
                       bool emit_energy_deposit_events, emit_fn &&emit) {
    for (particle_index_t particle_idx = 0;
         particle_idx < this->_particles.get_total_count(); ++particle_idx) {
      while (this->_particles.active(particle_idx)) {
        const auto primary_tag = this->_particles.get_primary_tag(particle_idx);
        const auto electron_tag =
            this->_particles.get_secondary_tag(particle_idx);

        const auto total_before = this->_particles.get_total_count();

        this->init(particle_idx);

        const bool track = this->_particles.next_scatter(particle_idx);
        const std::uint8_t scatter_type =
            track ? this->_particles.get_next_scatter(particle_idx) : 0;
        const particle before = this->_particles[particle_idx];

        this->intersect(particle_idx);
        this->scatter(particle_idx);

        const auto total_after = this->_particles.get_total_count();
        if (emit_secondary_spawn_events && total_after > total_before) {
          const int2 pix = pixels[primary_tag];
          for (particle_index_t new_idx = total_before; new_idx < total_after;
               ++new_idx) {
            const particle secondary = this->_particles[new_idx];
            interaction_event ev{};
            ev.x = static_cast<float>(secondary.pos.x);
            ev.y = static_cast<float>(secondary.pos.y);
            ev.z = static_cast<float>(secondary.pos.z);
            ev.dx = static_cast<float>(secondary.dir.x);
            ev.dy = static_cast<float>(secondary.dir.y);
            ev.dz = static_cast<float>(secondary.dir.z);
            ev.energy_before_ev = static_cast<float>(secondary.kin_energy);
            ev.energy_after_ev = ev.energy_before_ev;
            ev.delta_energy_ev = 0.0f;
            ev.primary_tag = static_cast<std::uint32_t>(primary_tag);
            ev.electron_tag =
                static_cast<std::uint32_t>(this->_particles.get_secondary_tag(
                    new_idx));
            ev.parent_electron_tag = static_cast<std::uint32_t>(electron_tag);
            ev.kind =
                static_cast<std::uint8_t>(interaction_kind::secondary_spawn);
            ev.scatter_type = 0;
            ev.px = pix.x;
            ev.py = pix.y;
            emit(ev);
          }
        }

        const particle after = this->_particles[particle_idx];
        if (emit_energy_deposit_events && track &&
            before.kin_energy != after.kin_energy) {
          const int2 pix = pixels[primary_tag];
          interaction_event ev{};
          ev.x = static_cast<float>(after.pos.x);
          ev.y = static_cast<float>(after.pos.y);
          ev.z = static_cast<float>(after.pos.z);
          ev.dx = static_cast<float>(after.dir.x);
          ev.dy = static_cast<float>(after.dir.y);
          ev.dz = static_cast<float>(after.dir.z);
          ev.energy_before_ev = static_cast<float>(before.kin_energy);
          ev.energy_after_ev = static_cast<float>(after.kin_energy);
          ev.delta_energy_ev =
              static_cast<float>(before.kin_energy - after.kin_energy);
          ev.primary_tag = static_cast<std::uint32_t>(primary_tag);
          ev.electron_tag = static_cast<std::uint32_t>(electron_tag);
          ev.parent_electron_tag = kInvalidElectronTag;
          ev.kind = static_cast<std::uint8_t>(interaction_kind::energy_deposit);
          ev.scatter_type = scatter_type;
          ev.px = pix.x;
          ev.py = pix.y;
          emit(ev);
        }
      }
    }
  }
};

void ResetProgress(simulation_progress *progress) {
  if (!progress) {
    return;
  }
  progress->primaries_remaining.store(0, std::memory_order_relaxed);
  progress->running_particles.store(0, std::memory_order_relaxed);
  progress->progress.store(0.0, std::memory_order_relaxed);
}

void FinishProgress(simulation_progress *progress) {
  if (!progress) {
    return;
  }
  progress->primaries_remaining.store(0, std::memory_order_relaxed);
  progress->running_particles.store(0, std::memory_order_relaxed);
  progress->progress.store(1.0, std::memory_order_relaxed);
}

}

bool run_simulation_streaming(
    const std::vector<triangle> &triangles,
    const cpu_material_manager<cpu_material_t> &materials,
    std::vector<particle> &primaries, std::vector<int2> &pixels,
    const simulation_settings &settings, interaction_callback_t on_event,
    void *user, simulation_stats *out_stats, std::string &out_error,
    simulation_progress *progress) {
  out_error.clear();

  if (out_stats) {
    *out_stats = {};
    out_stats->max_energy = materials.get_max_energy();
  }

  ResetProgress(progress);

  if (triangles.empty()) {
    out_error = "No triangles provided.";
    return false;
  }
  if (primaries.empty()) {
    out_error = "No primaries provided.";
    return false;
  }
  if (primaries.size() != pixels.size()) {
    out_error = "Primaries and pixel arrays are mismatched.";
    return false;
  }
  if (settings.primaries_per_batch == 0 || settings.event_batch_size == 0) {
    out_error = "Invalid simulation settings.";
    return false;
  }

  int max_material = -1;
  for (const triangle &tri : triangles) {
    if (tri.material_in >= 0) {
      max_material = (std::max)(max_material, tri.material_in);
    }
    if (tri.material_out >= 0) {
      max_material = (std::max)(max_material, tri.material_out);
    }
  }
  if (max_material >= 0 && materials.size() <= max_material) {
    out_error = "Not enough materials for provided geometry.";
    return false;
  }

  geometry_t geometry = geometry_t::create(triangles);

  std::vector<std::uint32_t> tags(primaries.size());
  std::iota(tags.begin(), tags.end(), 0);
  work_pool pool(primaries.data(), tags.data(), primaries.size());

  intersect_t inter;
  streaming_driver driver(inter, materials, geometry, settings.energy_threshold,
                          materials.get_max_energy(), settings.seed);

  std::vector<interaction_event> batch;
  batch.reserve(settings.event_batch_size);

  const std::size_t total_primaries = primaries.size();
  if (progress) {
    progress->primaries_remaining.store(total_primaries,
                                        std::memory_order_relaxed);
    progress->progress.store(0.0, std::memory_order_relaxed);
  }

  auto emit = [&](const interaction_event &ev) {
    if (out_stats) {
      out_stats->total_events += 1;
      switch (static_cast<interaction_kind>(ev.kind)) {
      case interaction_kind::primary_spawn:
        out_stats->primary_spawn_events += 1;
        break;
      case interaction_kind::secondary_spawn:
        out_stats->secondary_spawn_events += 1;
        break;
      case interaction_kind::energy_deposit:
        out_stats->energy_deposit_events += 1;
        break;
      }
    }

    if (!on_event) {
      return;
    }

    batch.push_back(ev);
    if (batch.size() >= settings.event_batch_size) {
      on_event(batch.data(), batch.size(), user);
      batch.clear();
    }
  };

  for (;;) {
    auto work_data = pool.get_work(settings.primaries_per_batch);
    const auto count = std::get<2>(work_data);
    if (count == 0) {
      break;
    }

    driver.push(std::get<0>(work_data), std::get<1>(work_data), count);

    if (settings.emit_primary_spawn_events) {
      const particle *new_primaries = std::get<0>(work_data);
      const std::uint32_t *new_tags = std::get<1>(work_data);
      for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t tag = new_tags[i];
        const int2 pix = pixels[tag];
        const particle p = new_primaries[i];

        interaction_event ev{};
        ev.x = static_cast<float>(p.pos.x);
        ev.y = static_cast<float>(p.pos.y);
        ev.z = static_cast<float>(p.pos.z);
        ev.dx = static_cast<float>(p.dir.x);
        ev.dy = static_cast<float>(p.dir.y);
        ev.dz = static_cast<float>(p.dir.z);
        ev.energy_before_ev = static_cast<float>(p.kin_energy);
        ev.energy_after_ev = ev.energy_before_ev;
        ev.delta_energy_ev = 0.0f;
        ev.primary_tag = tag;
        ev.electron_tag = 0;
        ev.parent_electron_tag = kInvalidElectronTag;
        ev.kind = static_cast<std::uint8_t>(interaction_kind::primary_spawn);
        ev.scatter_type = 0;
        ev.px = pix.x;
        ev.py = pix.y;
        emit(ev);
      }
    }

    driver.simulate_to_end(pixels, settings.emit_secondary_spawn_events,
                           settings.emit_energy_deposit_events, emit);

    driver.flush_detected([](particle const &, std::uint32_t) {});

    if (progress) {
      const auto remaining = pool.get_primaries_to_go();
      progress->primaries_remaining.store(remaining, std::memory_order_relaxed);
      double ratio = 1.0;
      if (total_primaries > 0) {
        ratio = 1.0 - (static_cast<double>(remaining) /
                       static_cast<double>(total_primaries));
      }
      progress->progress.store(ratio, std::memory_order_relaxed);
      progress->running_particles.store(0, std::memory_order_relaxed);
    }
  }

  if (on_event && !batch.empty()) {
    on_event(batch.data(), batch.size(), user);
    batch.clear();
  }

  geometry_t::destroy(geometry);

  FinishProgress(progress);

  return true;
}

bool run_simulation(const std::vector<triangle> &triangles,
                    const cpu_material_manager<cpu_material_t> &materials,
                    std::vector<particle> &primaries,
                    std::vector<int2> &pixels,
                    const simulation_settings &settings,
                    std::vector<interaction_event> &out_events,
                    simulation_stats *out_stats, std::string &out_error,
                    simulation_progress *progress) {
  out_events.clear();
  out_error.clear();

  auto on_event = [](const interaction_event *events, std::size_t count,
                     void *user) {
    auto *vec = static_cast<std::vector<interaction_event> *>(user);
    vec->insert(vec->end(), events, events + count);
  };

  return run_simulation_streaming(triangles, materials, primaries, pixels,
                                  settings, on_event, &out_events, out_stats,
                                  out_error, progress);
}

}

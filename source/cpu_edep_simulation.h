#ifndef __CPU_EDEP_SIMULATION_H_
#define __CPU_EDEP_SIMULATION_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "config/config.h"
#include "core/material.h"
#include "core/cpu_material_manager.h"
#include "core/particle.h"
#include "core/triangle.h"
#include "io/load_pri_file.h"
#include "physics_config.h"

namespace nbl::cpu_edep {

using cpu_material_t = material<scatter_physics<false>>;

enum class interaction_kind : std::uint8_t {
  primary_spawn = 0,
  secondary_spawn = 1,
  energy_deposit = 2,
  interface_crossing = 3,
};

struct interaction_event {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;

  float energy_before_ev = 0.0f;
  float energy_after_ev = 0.0f;
  float delta_energy_ev = 0.0f;

  std::uint32_t primary_tag = 0;
  std::uint32_t electron_tag = 0;
  std::uint32_t parent_electron_tag = 0xFFFFFFFFu;

  std::uint8_t kind = static_cast<std::uint8_t>(interaction_kind::energy_deposit);
  std::uint8_t scatter_type = 0;
  std::uint16_t _pad = 0;

  int px = 0;
  int py = 0;
};

using interaction_callback_t =
    void (*)(const interaction_event *events, std::size_t count, void *user);

struct simulation_settings {
  real energy_threshold = 0;
  unsigned int seed = 0x78c7e39b;

  std::uint32_t primaries_per_batch = 1;
  std::uint32_t event_batch_size = 4096;

  bool emit_primary_spawn_events = true;
  bool emit_secondary_spawn_events = true;
  bool emit_energy_deposit_events = true;
  bool emit_interface_events = false;
};

struct simulation_stats {
  std::uint64_t total_events = 0;
  std::uint64_t primary_spawn_events = 0;
  std::uint64_t secondary_spawn_events = 0;
  std::uint64_t energy_deposit_events = 0;
  real max_energy = 0;
};

struct simulation_progress {
  std::atomic<double> progress{0.0};
  std::atomic<std::uint64_t> primaries_remaining{0};
  std::atomic<std::uint32_t> running_particles{0};
};

bool run_simulation_streaming(
    const std::vector<triangle> &triangles,
    const cpu_material_manager<cpu_material_t> &materials,
    std::vector<particle> &primaries, std::vector<int2> &pixels,
    const simulation_settings &settings, interaction_callback_t on_event,
    void *user, simulation_stats *out_stats, std::string &out_error,
    simulation_progress *progress = nullptr);

bool run_simulation(const std::vector<triangle> &triangles,
                    const cpu_material_manager<cpu_material_t> &materials,
                    std::vector<particle> &primaries, std::vector<int2> &pixels,
                    const simulation_settings &settings,
                    std::vector<interaction_event> &out_events,
                    simulation_stats *out_stats, std::string &out_error,
                    simulation_progress *progress = nullptr);

} // namespace nbl::cpu_edep

#endif // __CPU_EDEP_SIMULATION_H_

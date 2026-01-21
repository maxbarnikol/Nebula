#include "gpu_simulation.h"

#include "common/work_pool.h"
#include "drivers/gpu/gpu_driver.h"
#include "geometry/octree.h"
#include "io/load_pri_file.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

namespace nbl::gpu {
namespace {

using geometry_t = nbl::geometry::octree<true>;
using driver = nbl::drivers::gpu_driver<scatter_physics<true>, intersect_t, geometry_t>;

struct worker_data {
  worker_data() : geometry(), primaries() {}

  std::condition_variable cv;
  std::mutex cv_m;

  nbl::geometry::octree_builder::linearized_octree geometry;
  const nbl::cpu_material_manager<cpu_material_t> *materials = nullptr;
  work_pool primaries;
  const std::vector<int2> *pixels = nullptr;
  std::vector<uint32_t> tags;

  uint32_t prescan_size = 0;
  real batch_factor = 0;

  uint32_t capacity = 0;
  uint32_t frame_size = 0;
  uint32_t batch_size = 0;
  real min_energy = 0;
  real max_energy = std::numeric_limits<real>::infinity();

  std::mutex running_mutex;
  std::vector<uint32_t> running_count;
  std::vector<std::vector<detected_electron>> detected;

  bool stream_detected = false;
  std::mutex detected_mutex;

  enum class status_t {
    init,
    geometry_loaded,
    materials_loaded,
    primaries_loaded,
    prescan_done
  };
  status_t status = status_t::init;
};

void worker_thread(worker_data &data, int gpu_id, typename driver::seed_t seed,
                   bool do_prescan) {
  cudaSetDevice(gpu_id);

  {
    std::unique_lock<std::mutex> lk(data.cv_m);
    data.cv.wait(lk, [&data]() {
      return data.status >= worker_data::status_t::geometry_loaded;
    });
  }
  geometry_t geometry = geometry_t::create(data.geometry);

  {
    std::unique_lock<std::mutex> lk(data.cv_m);
    data.cv.wait(lk, [&data]() {
      return data.status >= worker_data::status_t::materials_loaded;
    });
  }
  auto materials = nbl::gpu_material_manager<gpu_material_t>::create(*data.materials);

  intersect_t inter;
  driver d(data.capacity, inter, materials, geometry, data.min_energy,
           data.max_energy, seed);

  if (do_prescan) {
    {
      std::unique_lock<std::mutex> lk(data.cv_m);
      data.cv.wait(lk, [&data]() {
        return data.status >= worker_data::status_t::primaries_loaded;
      });
    }

    std::vector<std::pair<uint32_t, uint32_t>> prescan_stats;
    {
      auto work_data = data.primaries.get_work(data.prescan_size);
      auto particles_pushed = d.push(std::get<0>(work_data),
                                     std::get<1>(work_data),
                                     std::get<2>(work_data));
      prescan_stats.push_back({particles_pushed, 0});
    }

    while (prescan_stats.back().first > 0) {
      d.do_iteration();
      prescan_stats.push_back({d.get_running_count(), d.get_detected_count()});
    }

    const uint32_t frame_size =
        1 + static_cast<uint32_t>(std::distance(
                prescan_stats.begin(),
                std::max_element(prescan_stats.begin(), prescan_stats.end(),
                                 [](std::pair<uint32_t, uint32_t> p1,
                                    std::pair<uint32_t, uint32_t> p2) {
                                   return p1.first < p2.first;
                                 })));

    uint32_t batch_size = 0;
    {
      real accumulator = 0;
      for (uint32_t i = 2 * frame_size; i < prescan_stats.size();
           i += frame_size) {
        accumulator += prescan_stats[i].first / real(data.prescan_size);
      }
      accumulator += 2 * prescan_stats[frame_size].first / real(data.prescan_size);
      accumulator += 2 * prescan_stats[frame_size].second / real(data.prescan_size);
      batch_size = static_cast<uint32_t>(data.batch_factor * data.capacity /
                                         accumulator);
    }

    {
      std::lock_guard<std::mutex> lg(data.cv_m);
      data.frame_size = frame_size;
      data.batch_size = batch_size;
      data.status = worker_data::status_t::prescan_done;
    }
    data.cv.notify_all();
  } else {
    std::unique_lock<std::mutex> lk(data.cv_m);
    data.cv.wait(lk, [&data]() {
      return data.status >= worker_data::status_t::prescan_done;
    });
  }

  d.allocate_input_buffers(data.batch_size);
  d.push_to_buffer(data.primaries);

  if (!data.stream_detected) {
    std::vector<detected_electron> &out = data.detected[gpu_id];
    out.clear();

    for (;;) {
      d.push_to_simulation();
      d.buffer_detected();
      cudaDeviceSynchronize();

      for (uint32_t i = 0; i < data.frame_size; ++i) {
        d.do_iteration();
      }

      d.push_to_buffer(data.primaries);

      auto running_count =
          d.flush_buffered([&out, &data](particle p, uint32_t t) {
            detected_electron e{};
            e.x = static_cast<float>(p.pos.x);
            e.y = static_cast<float>(p.pos.y);
            e.z = static_cast<float>(p.pos.z);
            e.dx = static_cast<float>(p.dir.x);
            e.dy = static_cast<float>(p.dir.y);
            e.dz = static_cast<float>(p.dir.z);
            e.energy_ev = static_cast<float>(p.kin_energy);
            const int2 pix = (*data.pixels)[t];
            e.px = pix.x;
            e.py = pix.y;
            out.push_back(e);
          });
      {
        std::lock_guard<std::mutex> lock(data.running_mutex);
        data.running_count[gpu_id] = running_count;
      }

      cudaDeviceSynchronize();

      if (running_count == 0 && data.primaries.done()) {
        break;
      }
    }
  } else {
    {
      std::lock_guard<std::mutex> lock(data.detected_mutex);
      data.detected[gpu_id].clear();
    }

    std::vector<detected_electron> detected_batch;

    for (;;) {
      d.push_to_simulation();
      d.buffer_detected();
      cudaDeviceSynchronize();

      for (uint32_t i = 0; i < data.frame_size; ++i) {
        d.do_iteration();
      }

      d.push_to_buffer(data.primaries);

      detected_batch.clear();
      auto running_count =
          d.flush_buffered([&detected_batch, &data](particle p, uint32_t t) {
            detected_electron e{};
            e.x = static_cast<float>(p.pos.x);
            e.y = static_cast<float>(p.pos.y);
            e.z = static_cast<float>(p.pos.z);
            e.dx = static_cast<float>(p.dir.x);
            e.dy = static_cast<float>(p.dir.y);
            e.dz = static_cast<float>(p.dir.z);
            e.energy_ev = static_cast<float>(p.kin_energy);
            const int2 pix = (*data.pixels)[t];
            e.px = pix.x;
            e.py = pix.y;
            detected_batch.push_back(e);
          });

      if (!detected_batch.empty()) {
        std::lock_guard<std::mutex> lock(data.detected_mutex);
        auto &out = data.detected[gpu_id];
        out.insert(out.end(), detected_batch.begin(), detected_batch.end());
      }

      {
        std::lock_guard<std::mutex> lock(data.running_mutex);
        data.running_count[gpu_id] = running_count;
      }

      cudaDeviceSynchronize();

      if (running_count == 0 && data.primaries.done()) {
        break;
      }
    }
  }

  geometry_t::destroy(geometry);
  nbl::gpu_material_manager<gpu_material_t>::destroy(materials);
}

} // namespace

bool run_simulation(const std::vector<triangle> &triangles,
                    const cpu_material_manager<cpu_material_t> &materials,
                    std::vector<particle> &primaries,
                    std::vector<int2> &pixels,
                    const simulation_settings &settings,
                    std::vector<detected_electron> &out_detected,
                    simulation_stats *out_stats, std::string &out_error,
                    simulation_progress *progress) {
  out_error.clear();
  out_detected.clear();

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
  if (settings.capacity == 0 || settings.prescan_size == 0 ||
      !(settings.batch_factor > 0)) {
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

  int n_gpus = 0;
  cudaGetDeviceCount(&n_gpus);
  if (n_gpus <= 0) {
    out_error = "No CUDA devices found.";
    return false;
  }

  if (settings.sort_primaries) {
    nbl::sort_pri_file(primaries, pixels);
  }
  nbl::prescan_shuffle(primaries, pixels, settings.prescan_size);

  worker_data data{};
  data.capacity = settings.capacity;
  data.min_energy = settings.energy_threshold;
  data.max_energy = materials.get_max_energy();
  data.prescan_size = settings.prescan_size;
  data.batch_factor = settings.batch_factor;
  data.materials = &materials;
  data.pixels = &pixels;
  data.running_count.resize(n_gpus, 0);
  data.detected.resize(n_gpus);

  data.tags.resize(primaries.size());
  std::iota(data.tags.begin(), data.tags.end(), 0);
  data.primaries = work_pool(primaries.data(), data.tags.data(), primaries.size());

  data.geometry = nbl::geometry::octree_builder::linearized_octree(
      nbl::geometry::octree_builder::octree_root(triangles));

  {
    std::lock_guard<std::mutex> lg(data.cv_m);
    data.status = worker_data::status_t::geometry_loaded;
  }
  data.cv.notify_all();

  {
    std::lock_guard<std::mutex> lg(data.cv_m);
    data.status = worker_data::status_t::materials_loaded;
  }
  data.cv.notify_all();

  {
    std::lock_guard<std::mutex> lg(data.cv_m);
    data.status = worker_data::status_t::primaries_loaded;
  }
  data.cv.notify_all();

  std::mt19937 random_generator(settings.seed);
  std::vector<std::thread> threads;
  threads.reserve(n_gpus);
  for (int i = 0; i < n_gpus; ++i) {
    threads.emplace_back(worker_thread, std::ref(data), i, random_generator(),
                         i == 0);
  }

  {
    std::unique_lock<std::mutex> lk(data.cv_m);
    data.cv.wait(lk, [&data]() {
      return data.status >= worker_data::status_t::prescan_done;
    });
  }

  const std::size_t total_primaries = primaries.size();
  if (progress) {
    progress->primaries_remaining.store(total_primaries,
                                        std::memory_order_relaxed);
    progress->running_particles.store(0, std::memory_order_relaxed);
    progress->progress.store(0.0, std::memory_order_relaxed);
  }

  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto primaries_to_go = data.primaries.get_primaries_to_go();

    if (progress) {
      double ratio = 1.0;
      if (total_primaries > 0) {
        ratio = 1.0 - (static_cast<double>(primaries_to_go) /
                       static_cast<double>(total_primaries));
      }
      progress->primaries_remaining.store(primaries_to_go,
                                          std::memory_order_relaxed);
      progress->progress.store(ratio, std::memory_order_relaxed);

      uint32_t running = 0;
      {
        std::lock_guard<std::mutex> lock(data.running_mutex);
        for (const auto &count : data.running_count) {
          running += count;
        }
      }
      progress->running_particles.store(running, std::memory_order_relaxed);
    }

    if (primaries_to_go == 0) {
      bool any_running = false;
      {
        std::lock_guard<std::mutex> lock(data.running_mutex);
        for (const auto &count : data.running_count) {
          if (count != 0) {
            any_running = true;
            break;
          }
        }
      }
      if (!any_running) {
        break;
      }
    }
  }

  for (auto &t : threads) {
    t.join();
  }

  std::size_t total_detected = 0;
  for (const auto &vec : data.detected) {
    total_detected += vec.size();
  }
  out_detected.reserve(total_detected);
  for (auto &vec : data.detected) {
    out_detected.insert(out_detected.end(), vec.begin(), vec.end());
  }

  if (out_stats) {
    out_stats->frame_size = data.frame_size;
    out_stats->batch_size = data.batch_size;
    out_stats->gpu_count = static_cast<uint32_t>(n_gpus);
    out_stats->max_energy = data.max_energy;
  }

  if (progress) {
    progress->primaries_remaining.store(0, std::memory_order_relaxed);
    progress->running_particles.store(0, std::memory_order_relaxed);
    progress->progress.store(1.0, std::memory_order_relaxed);
  }

  return true;
}

bool run_simulation_streaming(const std::vector<triangle> &triangles,
                              const cpu_material_manager<cpu_material_t> &materials,
                              std::vector<particle> &primaries,
                              std::vector<int2> &pixels,
                              const simulation_settings &settings,
                              detected_callback_t on_detected, void *user,
                              simulation_stats *out_stats,
                              std::string &out_error,
                              simulation_progress *progress) {
  out_error.clear();

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
  if (settings.capacity == 0 || settings.prescan_size == 0 ||
      !(settings.batch_factor > 0)) {
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

  int n_gpus = 0;
  cudaGetDeviceCount(&n_gpus);
  if (n_gpus <= 0) {
    out_error = "No CUDA devices found.";
    return false;
  }

  if (settings.sort_primaries) {
    nbl::sort_pri_file(primaries, pixels);
  }
  nbl::prescan_shuffle(primaries, pixels, settings.prescan_size);

  worker_data data{};
  data.capacity = settings.capacity;
  data.min_energy = settings.energy_threshold;
  data.max_energy = materials.get_max_energy();
  data.prescan_size = settings.prescan_size;
  data.batch_factor = settings.batch_factor;
  data.materials = &materials;
  data.pixels = &pixels;
  data.running_count.resize(n_gpus, 0);
  data.detected.resize(n_gpus);
  data.stream_detected = true;

  data.tags.resize(primaries.size());
  std::iota(data.tags.begin(), data.tags.end(), 0);
  data.primaries = work_pool(primaries.data(), data.tags.data(), primaries.size());

  data.geometry = nbl::geometry::octree_builder::linearized_octree(
      nbl::geometry::octree_builder::octree_root(triangles));

  {
    std::lock_guard<std::mutex> lg(data.cv_m);
    data.status = worker_data::status_t::geometry_loaded;
  }
  data.cv.notify_all();

  {
    std::lock_guard<std::mutex> lg(data.cv_m);
    data.status = worker_data::status_t::materials_loaded;
  }
  data.cv.notify_all();

  {
    std::lock_guard<std::mutex> lg(data.cv_m);
    data.status = worker_data::status_t::primaries_loaded;
  }
  data.cv.notify_all();

  std::mt19937 random_generator(settings.seed);
  std::vector<std::thread> threads;
  threads.reserve(n_gpus);
  for (int i = 0; i < n_gpus; ++i) {
    threads.emplace_back(worker_thread, std::ref(data), i, random_generator(),
                         i == 0);
  }

  {
    std::unique_lock<std::mutex> lk(data.cv_m);
    data.cv.wait(lk, [&data]() {
      return data.status >= worker_data::status_t::prescan_done;
    });
  }

  const std::size_t total_primaries = primaries.size();
  if (progress) {
    progress->primaries_remaining.store(total_primaries,
                                        std::memory_order_relaxed);
    progress->running_particles.store(0, std::memory_order_relaxed);
    progress->progress.store(0.0, std::memory_order_relaxed);
  }

  std::vector<std::vector<detected_electron>> flush_buffers;
  flush_buffers.resize(n_gpus);

  auto flush_detected = [&]() {
    {
      std::lock_guard<std::mutex> lock(data.detected_mutex);
      for (int i = 0; i < n_gpus; ++i) {
        flush_buffers[i].swap(data.detected[i]);
      }
    }

    for (int i = 0; i < n_gpus; ++i) {
      auto &batch = flush_buffers[i];
      if (on_detected && !batch.empty()) {
        on_detected(batch.data(), batch.size(), user);
      }
      batch.clear();
    }
  };

  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto primaries_to_go = data.primaries.get_primaries_to_go();

    if (progress) {
      double ratio = 1.0;
      if (total_primaries > 0) {
        ratio = 1.0 - (static_cast<double>(primaries_to_go) /
                       static_cast<double>(total_primaries));
      }
      progress->primaries_remaining.store(primaries_to_go,
                                          std::memory_order_relaxed);
      progress->progress.store(ratio, std::memory_order_relaxed);

      uint32_t running = 0;
      {
        std::lock_guard<std::mutex> lock(data.running_mutex);
        for (const auto &count : data.running_count) {
          running += count;
        }
      }
      progress->running_particles.store(running, std::memory_order_relaxed);
    }

    flush_detected();

    if (primaries_to_go == 0) {
      bool any_running = false;
      {
        std::lock_guard<std::mutex> lock(data.running_mutex);
        for (const auto &count : data.running_count) {
          if (count != 0) {
            any_running = true;
            break;
          }
        }
      }
      if (!any_running) {
        break;
      }
    }
  }

  for (auto &t : threads) {
    t.join();
  }

  flush_detected();

  if (out_stats) {
    out_stats->frame_size = data.frame_size;
    out_stats->batch_size = data.batch_size;
    out_stats->gpu_count = static_cast<uint32_t>(n_gpus);
    out_stats->max_energy = data.max_energy;
  }

  if (progress) {
    progress->primaries_remaining.store(0, std::memory_order_relaxed);
    progress->running_particles.store(0, std::memory_order_relaxed);
    progress->progress.store(1.0, std::memory_order_relaxed);
  }

  return true;
}

} // namespace nbl::gpu

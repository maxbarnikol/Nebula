#ifndef __GPU_SIMULATION_H_
#define __GPU_SIMULATION_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "config/config.h"
#include "core/material.h"
#include "core/cpu_material_manager.h"
#include "core/particle.h"
#include "core/triangle.h"
#include "io/load_pri_file.h"
#include "physics_config.h"

namespace nbl::gpu {

using cpu_material_t = material<scatter_physics<false>>;
using gpu_material_t = material<scatter_physics<true>>;

struct detected_electron {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  float energy_ev = 0.0f;
  int px = 0;
  int py = 0;
};

static_assert(sizeof(detected_electron) == 7 * sizeof(float) + 2 * sizeof(int));

struct simulation_settings {
  real energy_threshold = 0;
  uint32_t capacity = 1000000;
  uint32_t prescan_size = 1000;
  real batch_factor = 0.9;
  unsigned int seed = 0x78c7e39b;
  bool sort_primaries = false;
};

struct simulation_stats {
  uint32_t frame_size = 0;
  uint32_t batch_size = 0;
  uint32_t gpu_count = 0;
  real max_energy = 0;
};

struct simulation_progress {
  std::atomic<double> progress{0.0};
  std::atomic<uint64_t> primaries_remaining{0};
  std::atomic<uint32_t> running_particles{0};
};

// Runs the GPU simulation. The primaries and pixels arrays are modified in-place
// when sort_primaries or prescan shuffling are enabled.
bool run_simulation(const std::vector<triangle> &triangles,
                    const cpu_material_manager<cpu_material_t> &materials,
                    std::vector<particle> &primaries,
                    std::vector<int2> &pixels,
                    const simulation_settings &settings,
                    std::vector<detected_electron> &out_detected,
                    simulation_stats *out_stats, std::string &out_error,
                    simulation_progress *progress = nullptr);

} // namespace nbl::gpu

#endif // __GPU_SIMULATION_H_

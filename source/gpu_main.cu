#include "gpu_simulation.h"

#include "common/cli_params.h"
#include "geometry/octree.h"
#include "io/load_pri_file.h"
#include "io/load_tri_file.h"
#include "io/output_stream.h"
#include "physics_config.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

int main(int argc, char **argv) {
  std::clog << "This is Nebula version " << VERSION_MAJOR << "."
            << VERSION_MINOR << "." << VERSION_PATCH << "\n\n"
            << "Physics models:\n";
  scatter_physics<true>::print_info(std::clog);
  intersect_t::print_info(std::clog);
  std::clog << "\n" << std::string(80, '-') << "\n\n";

  cli_params p("[options] <geometry.tri> <primaries.pri> [material0.mat] .. [materialN.mat]");
  p.add_option("energy-threshold", "Lowest energy to simulate", 0);
  p.add_option("capacity", "Electron capacity on the GPU", 1000000);
  p.add_option("prescan-size", "Number of electrons to use for prescan", 1000);
  p.add_option("batch-factor", "Multiplication factor for electron batch size", 0.9);
  p.add_option("seed", "Random seed", 0x78c7e39b);
  p.add_option("sort-primaries", "Sort primary electrons before simulation", false);
  p.parse(argc, argv);

  const real energy_threshold = p.get_flag<real>("energy-threshold");
  const uint32_t capacity = p.get_flag<uint32_t>("capacity");
  const uint32_t prescan_size = p.get_flag<uint32_t>("prescan-size");
  const real batch_factor = p.get_flag<real>("batch-factor");
  const unsigned int seed = p.get_flag<unsigned int>("seed");
  const bool sort_primaries = p.get_flag<bool>("sort-primaries");

  std::vector<std::string> pos_flags = p.get_positional();
  if (pos_flags.size() < 3 || capacity <= 0 || prescan_size <= 0 ||
      batch_factor <= 0) {
    p.print_usage(std::clog);
    return 1;
  }

  std::vector<triangle> triangles = nbl::load_tri_file(pos_flags[0]);
  if (triangles.empty()) {
    std::clog << "Error: could not load triangles!\n";
    p.print_usage(std::clog);
    return 1;
  }

  nbl::cpu_material_manager<nbl::gpu::cpu_material_t> materials;
  for (size_t parameter_idx = 2; parameter_idx < pos_flags.size();
       ++parameter_idx) {
    nbl::hdf5_file material(pos_flags[parameter_idx]);
    materials.add(material);

    std::clog << "  Material " << (parameter_idx - 2) << ":\n"
              << "    Name: " << material.get_property_string("name") << "\n"
              << "    cstool version: "
              << material.get_property_string("cstool_version") << "\n";
  }

  nbl::geometry::octree_builder::linearized_octree geometry{
      nbl::geometry::octree_builder::octree_root(triangles)};

  std::vector<particle> primaries;
  std::vector<int2> pixels;
  std::tie(primaries, pixels) = nbl::load_pri_file(
      pos_flags[1], geometry.AABB_min(), geometry.AABB_max(),
      materials.get_max_energy());
  if (primaries.empty()) {
    std::clog << "Error: could not load primary electrons!\n";
    p.print_usage(std::clog);
    return 1;
  }

  nbl::gpu::simulation_settings settings{};
  settings.energy_threshold = energy_threshold;
  settings.capacity = capacity;
  settings.prescan_size = prescan_size;
  settings.batch_factor = batch_factor;
  settings.seed = seed;
  settings.sort_primaries = sort_primaries;

  nbl::gpu::simulation_stats stats{};
  nbl::gpu::simulation_progress progress{};
  std::vector<nbl::gpu::detected_electron> detected;
  std::string error;

  const bool ok = nbl::gpu::run_simulation(triangles, materials, primaries, pixels,
                                           settings, detected, &stats, error,
                                           &progress);
  if (!ok) {
    std::clog << "Error: " << error << "\n";
    return 1;
  }

  output_stream out("stdout");
  output_buffer buff(out, 1024 * (7 * sizeof(float) + 2 * sizeof(int)));
  for (const auto &e : detected) {
    buff.add(std::array<float, 7>{e.x, e.y, e.z, e.dx, e.dy, e.dz, e.energy_ev});
    buff.add(std::array<int, 2>{e.px, e.py});
  }
  buff.flush();

  cudaError_t err = cudaDeviceSynchronize();
  if (err == 0) {
    std::clog << "\nSimulation successful!\n\n";
  } else {
    std::clog << "\nSimulation ended with CUDA error code " << err << "\n\n";
  }

  std::clog << "Detected electrons: " << detected.size() << "\n";
  std::clog << "frame_size = " << stats.frame_size
            << " | batch_size = " << stats.batch_size << "\n";

  return 0;
}

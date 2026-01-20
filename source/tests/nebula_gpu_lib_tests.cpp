#include "gpu_simulation.h"

#include "io/hdf5_file.h"
#include "io/load_pri_file.h"
#include "io/load_tri_file.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {

bool Expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << std::endl;
    return false;
  }
  std::cout << "PASSED: " << message << std::endl;
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: nebula_gpu_lib_tests <testdata_dir>" << std::endl;
    return 1;
  }

  int gpu_count = 0;
  cudaGetDeviceCount(&gpu_count);
  if (gpu_count <= 0) {
    std::cout << "Skipping Nebula GPU test: no CUDA devices found." << std::endl;
    return 0;
  }

  std::filesystem::path testdata = argv[1];
  const auto tri_path = testdata / "test" / "geo.tri";
  const auto pri_path = testdata / "test" / "beam.pri";
  const auto pmma_path = testdata / "pmma.mat";
  const auto silicon_path = testdata / "silicon.mat";

  if (!Expect(std::filesystem::exists(tri_path), "Geometry file exists") ||
      !Expect(std::filesystem::exists(pri_path), "Beam file exists") ||
      !Expect(std::filesystem::exists(pmma_path), "PMMA material exists") ||
      !Expect(std::filesystem::exists(silicon_path), "Silicon material exists")) {
    return 1;
  }

  std::vector<triangle> triangles =
      nbl::load_tri_file(tri_path.string());
  if (!Expect(!triangles.empty(), "Loaded triangles")) {
    return 1;
  }

  nbl::cpu_material_manager<nbl::gpu::cpu_material_t> materials;
  materials.add(nbl::hdf5_file(pmma_path.string()));
  materials.add(nbl::hdf5_file(silicon_path.string()));

  vec3 aabb_min = triangles[0].AABB_min();
  vec3 aabb_max = triangles[0].AABB_max();
  for (const auto &tri : triangles) {
    const vec3 tri_min = tri.AABB_min();
    const vec3 tri_max = tri.AABB_max();
    aabb_min = {std::min(aabb_min.x, tri_min.x),
                std::min(aabb_min.y, tri_min.y),
                std::min(aabb_min.z, tri_min.z)};
    aabb_max = {std::max(aabb_max.x, tri_max.x),
                std::max(aabb_max.y, tri_max.y),
                std::max(aabb_max.z, tri_max.z)};
  }

  std::vector<particle> primaries;
  std::vector<int2> pixels;
  std::tie(primaries, pixels) = nbl::load_pri_file(
      pri_path.string(), aabb_min, aabb_max, materials.get_max_energy());

  if (!Expect(!primaries.empty(), "Loaded primaries")) {
    return 1;
  }

  nbl::gpu::simulation_settings settings{};
  settings.capacity = 1000000;
  settings.prescan_size = 1000;
  settings.batch_factor = 0.9;
  settings.seed = 0x78c7e39b;
  settings.sort_primaries = false;

  nbl::gpu::simulation_stats stats{};
  nbl::gpu::simulation_progress progress{};
  std::vector<nbl::gpu::detected_electron> detected;
  std::string error;

  const bool ok = nbl::gpu::run_simulation(triangles, materials, primaries,
                                           pixels, settings, detected, &stats,
                                           error, &progress);
  if (!Expect(ok, "Simulation completes")) {
    std::cerr << "Error: " << error << std::endl;
    return 1;
  }

  if (!Expect(!detected.empty(), "Detected electrons are non-empty")) {
    return 1;
  }
  if (!Expect(stats.frame_size > 0, "Frame size computed")) {
    return 1;
  }
  if (!Expect(stats.batch_size > 0, "Batch size computed")) {
    return 1;
  }

  std::cout << "Detected electrons: " << detected.size() << std::endl;
  std::cout << "frame_size = " << stats.frame_size
            << " | batch_size = " << stats.batch_size << std::endl;

  return 0;
}

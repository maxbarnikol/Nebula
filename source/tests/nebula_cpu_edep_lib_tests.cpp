#include "cpu_edep_simulation.h"

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

}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: nebula_cpu_edep_lib_tests <testdata_dir>" << std::endl;
    return 1;
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

  std::vector<triangle> triangles = nbl::load_tri_file(tri_path.string());
  if (!Expect(!triangles.empty(), "Loaded triangles")) {
    return 1;
  }

  nbl::cpu_material_manager<nbl::cpu_edep::cpu_material_t> materials;
  materials.add(nbl::hdf5_file(pmma_path.string()));
  materials.add(nbl::hdf5_file(silicon_path.string()));

  vec3 aabb_min = triangles[0].AABB_min();
  vec3 aabb_max = triangles[0].AABB_max();
  for (const auto &tri : triangles) {
    const vec3 tri_min = tri.AABB_min();
    const vec3 tri_max = tri.AABB_max();
    aabb_min = {std::min(aabb_min.x, tri_min.x), std::min(aabb_min.y, tri_min.y),
                std::min(aabb_min.z, tri_min.z)};
    aabb_max = {std::max(aabb_max.x, tri_max.x), std::max(aabb_max.y, tri_max.y),
                std::max(aabb_max.z, tri_max.z)};
  }

  std::vector<particle> primaries;
  std::vector<int2> pixels;
  std::tie(primaries, pixels) = nbl::load_pri_file(pri_path.string(), aabb_min,
                                                   aabb_max,
                                                   materials.get_max_energy());

  if (!Expect(!primaries.empty(), "Loaded primaries")) {
    return 1;
  }

  constexpr std::size_t kMaxPrimaries = 32;
  if (primaries.size() > kMaxPrimaries) {
    primaries.resize(kMaxPrimaries);
    pixels.resize(kMaxPrimaries);
  }

  nbl::cpu_edep::simulation_settings settings{};
  settings.energy_threshold = 0;
  settings.seed = 0x78c7e39b;
  settings.primaries_per_batch = 1;
  settings.event_batch_size = 4096;
  settings.emit_primary_spawn_events = true;
  settings.emit_secondary_spawn_events = true;
  settings.emit_energy_deposit_events = true;

  nbl::cpu_edep::simulation_stats stats{};
  nbl::cpu_edep::simulation_progress progress{};
  std::vector<nbl::cpu_edep::interaction_event> events;
  std::string error;

  const bool ok = nbl::cpu_edep::run_simulation(
      triangles, materials, primaries, pixels, settings, events, &stats, error,
      &progress);
  if (!Expect(ok, "Simulation completes")) {
    std::cerr << "Error: " << error << std::endl;
    return 1;
  }

  if (!Expect(!events.empty(), "Interaction events are non-empty")) {
    return 1;
  }
  if (!Expect(stats.primary_spawn_events == primaries.size(),
              "Primary spawn event count matches primaries")) {
    return 1;
  }
  if (!Expect(stats.energy_deposit_events > 0, "Energy deposit events exist")) {
    return 1;
  }
  if (!Expect(progress.progress.load(std::memory_order_relaxed) >= 1.0,
              "Progress reaches 100%")) {
    return 1;
  }

  std::cout << "Primaries simulated: " << primaries.size() << std::endl;
  std::cout << "Events: total=" << stats.total_events
            << " | primary_spawn=" << stats.primary_spawn_events
            << " | secondary_spawn=" << stats.secondary_spawn_events
            << " | energy_deposit=" << stats.energy_deposit_events << std::endl;

  return 0;
}

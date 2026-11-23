#include "fragment_size_theory.hh"
#include "make_bar.hh"
#include "material_field.hh"
#include "parameters.hh"
#include "simulate_bar.hh"

#include <iostream>
#include <numeric>
#include <vector>

int main() {
  // --------------------------------------------------------------------------
  // Read simulation parameters.

  const Parameters params = readParameters("../src/PD/parameters.dat");

  // --------------------------------------------------------------------------
  // Load the common physical material realisation.

  const MaterialField material_field =
      MaterialField::load("../src/common/material_field.dat");

  // --------------------------------------------------------------------------
  // Construct the peridynamic bar using the common material field.

  Bar bar = makeBar(params, material_field);

  // --------------------------------------------------------------------------
  // Basic model information.

  std::cout << "Bar: " << bar.num_points << " points, " << bar.bond.count
            << " bonds\n";

  std::cout << "dx: " << bar.dx * 1e6 << " um\n";

  std::cout << "Horizon: " << bar.horizon * 1e6 << " um\n";

  std::cout << "m = horizon/dx: " << bar.m << "\n";

  std::cout << "Time step: " << bar.dt << " s\n";

  std::cout << "Early stopping: "
            << (params.enable_early_stopping ? "enabled" : "disabled") << "\n";

  // --------------------------------------------------------------------------
  // Run simulation.

  SimulationResults results = simulateBar(params, bar);

  // --------------------------------------------------------------------------
  // Analytical fragment-size predictions.

  const FragmentSizeTheory theory = fragmentSizeTheory(params);

  // --------------------------------------------------------------------------
  // Final mean fragment size.

  double average_size = 0.0;

  if (!results.sizes.empty()) {
    average_size =
        std::accumulate(results.sizes.begin(), results.sizes.end(), 0.0) /
        static_cast<double>(results.sizes.size());
  }

  // --------------------------------------------------------------------------
  // Final terminal summary.

  std::cout
      << "\n------------------------------------------------------------\n";

  std::cout << "Fragments: " << results.num_fragments << "\n";

  std::cout << "Raw fragments: " << results.raw_num_fragments << "\n";

  std::cout << "Average fragment size: " << average_size * 1e6 << " um\n";

  std::cout << "ZMR prediction: " << theory.zmr * 1e6 << " um\n";

  std::cout << "Grady prediction: " << theory.grady * 1e6 << " um\n";

  std::cout << "Final kinetic energy: " << results.kinetic_energy << " J\n";

  std::cout << "Final elastic energy: " << results.elastic_energy << " J\n";

  std::cout << "Final fracture energy: " << results.fracture_energy << " J\n";

  std::cout << "External work: " << results.external_work << " J\n";

  std::cout << "Energy error: " << results.energy_error << " J\n";

  std::cout << "Maximum relative energy error: "
            << results.max_relative_energy_error << "\n";

  std::cout << "Steps: " << results.steps << "\n";

  std::cout << "Final physical time: " << results.physical_time << " s\n";

  std::cout << "Simulation wall time: " << results.simulation_wall_time
            << " s\n";

  std::cout << "Total wall time: " << results.wall_time << " s\n";

  std::cout << "Stopping reason: " << results.stopping_reason << "\n";

  return 0;
}
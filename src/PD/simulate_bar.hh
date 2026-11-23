#ifndef SIMULATE_BAR_HH
#define SIMULATE_BAR_HH

#include "bar.hh"

#include <cstddef>
#include <string>
#include <vector>

struct Parameters;

/**
 * @brief Final and summary information returned by one PD simulation.
 *
 * Time-dependent diagnostics and spatial field snapshots are written directly
 * to disk during the simulation. Only final-state information needed by the
 * caller is retained here.
 */
struct SimulationResults {
  // ------------------------------------------------------------------------ //
  // Simulation summary

  /// Number of completed time steps.
  std::size_t steps;

  /// Final physical simulation time [s].
  double physical_time;

  /// Reason why the simulation stopped.
  std::string stopping_reason;

  /// Total wall-clock time of simulateBar(), including output operations [s].
  double wall_time;

  /// Cumulative computational time spent advancing the simulation,
  /// excluding diagnostic/output operations [s].
  double simulation_wall_time;

  // ------------------------------------------------------------------------ //
  // Final energy state

  /// Final kinetic energy [J].
  double kinetic_energy;

  /// Final recoverable peridynamic bond energy [J].
  double elastic_energy;

  /// Final dissipated fracture energy [J].
  double fracture_energy;

  /// Final work supplied by the prescribed boundary motion [J].
  double external_work;

  /// Final total energy used in the energy balance [J].
  double total_energy;

  /// Final absolute energy-balance residual [J].
  double energy_error;

  /// Maximum relative energy-balance error observed during the run [-].
  double max_relative_energy_error;

  /// Final total linear momentum of the physical bar [kg m/s].
  double momentum;

  // ------------------------------------------------------------------------ //
  // Final filtered fragment state

  /// Final number of fragments after applying the fragment filter.
  std::size_t num_fragments;

  /// Final filtered cut positions [m].
  std::vector<double> cuts;

  /// Final filtered fragment sizes [m].
  std::vector<double> sizes;

  // ------------------------------------------------------------------------ //
  // Final raw fragment topology

  /// Final number of fragments before fragment-size filtering.
  std::size_t raw_num_fragments;

  /// Final unfiltered cut positions [m].
  std::vector<double> raw_cuts;

  /// Final unfiltered fragment sizes [m].
  std::vector<double> raw_sizes;

  // ------------------------------------------------------------------------ //
  // Final bond state
  //
  // These arrays are retained because they are useful for optional detailed
  // post-processing without storing their complete time histories.

  /// Final health of every bond [-].
  std::vector<double> bond_health;

  /// Maximum historical stretch reached by every bond [-].
  std::vector<double> bond_max_stretch;

  /// Cumulative fracture energy dissipated by every bond [J].
  std::vector<double> bond_dissipated;
};

/**
 * @brief Simulates the dynamic fragmentation of the peridynamic bar.
 *
 * The bar begins unstrained with the centred velocity field
 *
 * @f[
 * v(x) = \dot{\varepsilon}x,
 * @f]
 *
 * corresponding to a uniform strain rate and zero total momentum. Boundary 
 * points continue to follow this prescribed motion throughout the simulation.
 *
 * Time integration is performed using velocity-Verlet. At each time step the
 * bond stretches are computed, irreversible bond damage is updated, forces are
 * recalculated, and the point velocities are advanced.
 *
 * Time-dependent global diagnostics, point fields, fragment information,
 * connectivity fields and fracture-event information are written directly to
 * disk during the simulation rather than being stored in memory.
 *
 * The simulation stops either when the prescribed maximum simulation time is
 * reached or when the filtered number of fragments remains unchanged for the
 * prescribed number of consecutive fragment-count checks.
 *
 * @param params Simulation and material parameters.
 * @param bar Initialised peridynamic bar.
 *
 * @return Final-state and summary information for the simulation.
 */
SimulationResults simulateBar(const Parameters& params, const Bar& bar);

#endif
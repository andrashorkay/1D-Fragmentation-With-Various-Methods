#ifndef PD_DIAGNOSTIC_WRITER_HH
#define PD_DIAGNOSTIC_WRITER_HH

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

struct Bar;
struct Parameters;

/**
 * @brief Records a change in the damage state of one peridynamic bond.
 *
 * Two event types are currently recorded:
 *
 * - "softening_started": the bond health first falls below 1;
 * - "fully_broken": the bond health first reaches 0.
 */
struct BondEvent {
  std::size_t step;
  double time;

  std::size_t bond_id;
  std::size_t i;
  std::size_t j;

  double x_i;
  double x_j;
  double x_mid;
  double length;

  std::string event;

  double stretch;
  double max_stretch;
  double health;

  double s_elastic;
  double s_break;

  double dissipated_increment;
};

/**
 * @brief Opens and stores the persistent PD output streams.
 *
 * These files remain open throughout the simulation so that global
 * diagnostics, fragment data and bond events can be appended without
 * repeatedly opening and closing the files.
 */
struct PDOutputFiles {
  std::ofstream* diagnostics;
  std::ofstream* fragments;
  std::ofstream* raw_fragments;
  std::ofstream* pd_diagnostics;
  std::ofstream* bond_events;
};

/**
 * @brief Creates output directories and initialises all persistent output
 * files used during the PD simulation.
 *
 * The following files are created:
 *
 * - diagnostics.out
 * - fragments.out
 * - raw_fragments.out
 * - pd_diagnostics.out
 * - bond_events.out
 *
 * The function also creates the directories used for spatial snapshots.
 *
 * @param params Simulation parameters containing the output directory and
 * file names.
 *
 * @return Open output streams used during the simulation.
 */
PDOutputFiles initialisePDOutputs(const Parameters& params);

/**
 * @brief Closes and releases all persistent PD output streams.
 *
 * @param files Output streams returned by initialisePDOutputs().
 */
void closePDOutputs(PDOutputFiles& files);

/**
 * @brief Writes static simulation and peridynamic-model metadata.
 *
 * This contains quantities that do not evolve during the simulation, such
 * as geometry, material parameters, discretisation, horizon, number of
 * bonds, bond-law parameters and output frequencies.
 *
 * @param params Simulation parameters.
 * @param bar Initialised peridynamic bar.
 */
void writePDMetadata(const Parameters& params, const Bar& bar);

/**
 * @brief Writes the spatially varying material properties once.
 *
 * For every physical material point, the file contains its reference
 * position, local strength and local toughness.
 *
 * @param params Simulation parameters.
 * @param bar Initialised peridynamic bar.
 */
void writePDMaterialProperties(const Parameters& params, const Bar& bar);

/**
 * @brief Writes all bond-damage events generated during one time step.
 *
 * @param output Bond-event output stream.
 * @param events Events detected during the current time step.
 */
void writePDBondEvents(std::ofstream& output,
                       const std::vector<BondEvent>& events);

/**
 * @brief Writes one complete PD diagnostic snapshot.
 *
 * This produces:
 *
 * - one row in the common CZM-compatible diagnostics file;
 * - filtered fragment information;
 * - raw fragment information;
 * - PD-specific global diagnostics;
 * - point-field data;
 * - gap-connectivity data.
 *
 * Spatial fields are written directly to disk rather than retained in
 * SimulationResults.
 *
 * @param params Simulation parameters.
 * @param bar Peridynamic bar.
 * @param step Current completed time step.
 * @param time Current physical time [s].
 * @param rate Imposed strain rate [1/s].
 * @param u Current displacement of all points [m].
 * @param v Current velocity of all points [m/s].
 * @param acceleration Current acceleration of physical points [m/s^2].
 * @param force Current peridynamic force density of all points.
 * @param health Current health of every bond.
 * @param stretch Current stretch of every bond.
 * @param max_stretch Maximum historical stretch of every bond.
 * @param dissipated Cumulative fracture energy [J].
 * @param external_work Cumulative external work [J].
 * @param initial_total_energy Initial total system energy [J].
 * @param simulation_wall_time Cumulative compute-only wall time [s].
 * @param simulation_wall_time_since_last_dump Compute-only wall time since
 * the previous dump [s].
 * @param new_softening_since_last_dump Number of bonds that began softening
 * since the previous dump.
 * @param new_broken_since_last_dump Number of bonds that fully broke since
 * the previous dump.
 * @param initial_active_crossing_bonds Initial number of active bonds
 * crossing each gap.
 * @param initial_health_connectivity Initial health-weighted connectivity
 * of each gap.
 * @param files Persistent output streams.
 *
 * @return Relative absolute energy-balance error at the current state.
 */
double
writePDSimulationDump(const Parameters& params,
                      const Bar& bar,
                      std::size_t step,
                      double time,
                      double rate,
                      const std::vector<double>& u,
                      const std::vector<double>& v,
                      const std::vector<double>& acceleration,
                      const std::vector<double>& force,
                      const std::vector<double>& health,
                      const std::vector<double>& stretch,
                      const std::vector<double>& max_stretch,
                      double dissipated,
                      double external_work,
                      double initial_total_energy,
                      double simulation_wall_time,
                      double simulation_wall_time_since_last_dump,
                      std::size_t new_softening_since_last_dump,
                      std::size_t new_broken_since_last_dump,
                      const std::vector<long>& initial_active_crossing_bonds,
                      const std::vector<double>& initial_health_connectivity,
                      PDOutputFiles& files);

/**
 * @brief Writes the final state of every damaged breakable bond.
 *
 * Intact bonds that never dissipated fracture energy are omitted to avoid
 * producing an unnecessarily large output file.
 *
 * @param params Simulation parameters.
 * @param bar Peridynamic bar.
 * @param u Final displacement field.
 * @param health Final health of every bond.
 * @param stretch Final stretch of every bond.
 * @param max_stretch Maximum historical stretch of every bond.
 * @param bond_dissipated Cumulative dissipated energy of every bond [J].
 */
void writePDFinalDamagedBonds(const Parameters& params,
                              const Bar& bar,
                              const std::vector<double>& u,
                              const std::vector<double>& health,
                              const std::vector<double>& stretch,
                              const std::vector<double>& max_stretch,
                              const std::vector<double>& bond_dissipated);

#endif
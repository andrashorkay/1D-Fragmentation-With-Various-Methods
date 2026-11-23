#include "pd_diagnostic_writer.hh"

#include "bar.hh"
#include "count_fragments.hh"
#include "parameters.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ============================================================================
// Internal data structures
// ============================================================================

struct PointFields {
  std::vector<double> damage;
  std::vector<double> strain;
  std::vector<double> stress;
};

struct BoundaryReactions {
  double left;
  double right;
};

struct FragmentStatistics {
  double mean_size;
  double min_size;
  double max_size;
};

struct FieldStatistics {
  double average_stress;
  double max_stress;
  double average_strain;
  double max_strain;
  double mean_damage;
  double max_damage;
};

struct BondStatistics {
  std::size_t num_intact;
  std::size_t num_softening;
  std::size_t num_broken;

  double fraction_broken;
  double mean_health;
  double min_health;

  double max_current_stretch;
  double max_historical_stretch;
};

struct ElasticEnergies {
  double total;
  double physical;
  double breakable;
};

// ============================================================================
// Point fields
// ============================================================================

PointFields computePointFields(const std::vector<double>& u,
                               const std::vector<double>& health,
                               const std::vector<double>& stretch,
                               const Bar& bar) {
  PointFields fields;

  fields.damage.resize(bar.num_points, 0.0);
  fields.strain.resize(bar.num_points, 0.0);
  fields.stress.resize(bar.num_points, 0.0);

  std::vector<double> damage_numerator(bar.num_points, 0.0);
  std::vector<double> damage_denominator(bar.num_points, 0.0);

  std::vector<double> strain_numerator(bar.num_points, 0.0);
  std::vector<double> strain_denominator(bar.num_points, 0.0);

  const std::size_t first_inside = bar.num_boundary;
  const std::size_t last_inside = bar.num_boundary + bar.num_points;

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    const std::size_t i = bar.bond.i[b];
    const std::size_t j = bar.bond.j[b];

    // ----------------------------------------------------------------------
    // Point damage.
    //
    // Only breakable bonds are included. This prevents the permanently
    // intact ghost-boundary bonds from artificially reducing damage near
    // the ends of the physical bar.

    if (bar.bond.can_break[b]) {
      const double weight = bar.volume * bar.bond.overlap[b];

      const std::size_t pi = i - first_inside;
      const std::size_t pj = j - first_inside;

      damage_numerator[pi] += health[b] * weight;
      damage_numerator[pj] += health[b] * weight;

      damage_denominator[pi] += weight;
      damage_denominator[pj] += weight;
    }

    // ----------------------------------------------------------------------
    // Local strain.
    //
    // A weighted displacement-gradient fit is used. The bond health is
    // included in the weight so fully broken bonds do not interpret crack
    // opening as bulk strain.

    const double strain_weight = health[b] * bar.bond.k[b];

    if (strain_weight > 0.0) {
      const double length = bar.bond.length[b];
      const double du = u[j] - u[i];

      const double numerator = strain_weight * length * du;

      const double denominator = strain_weight * length * length;

      if (i >= first_inside && i < last_inside) {
        const std::size_t p = i - first_inside;

        strain_numerator[p] += numerator;
        strain_denominator[p] += denominator;
      }

      if (j >= first_inside && j < last_inside) {
        const std::size_t p = j - first_inside;

        strain_numerator[p] += numerator;
        strain_denominator[p] += denominator;
      }
    }

    // ----------------------------------------------------------------------
    // 1D peridynamic stress.

    const double stress_contribution =
        0.5 * health[b] * bar.bond.k[b] * stretch[b] * bar.bond.length[b];

    if (i >= first_inside && i < last_inside) {
      fields.stress[i - first_inside] += stress_contribution;
    }

    if (j >= first_inside && j < last_inside) {
      fields.stress[j - first_inside] += stress_contribution;
    }
  }

  const double eps = std::numeric_limits<double>::epsilon();

  for (std::size_t p = 0; p < bar.num_points; ++p) {
    if (damage_denominator[p] > eps) {
      fields.damage[p] = 1.0 - damage_numerator[p] / damage_denominator[p];
    }

    if (strain_denominator[p] > eps) {
      fields.strain[p] = strain_numerator[p] / strain_denominator[p];
    } else {
      fields.strain[p] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  return fields;
}

// ============================================================================
// Global mechanics diagnostics
// ============================================================================

double
computeKineticEnergy(const std::vector<double>& v, const Bar& bar, double rho) {
  double kinetic = 0.0;

  for (const std::size_t i : bar.inside) {
    kinetic += v[i] * v[i];
  }

  return 0.5 * rho * bar.volume * kinetic;
}

ElasticEnergies computeElasticEnergies(const std::vector<double>& health,
                                       const std::vector<double>& stretch,
                                       const Bar& bar) {
  ElasticEnergies energies{0.0, 0.0, 0.0};

  const std::size_t first_inside = bar.num_boundary;
  const std::size_t last_inside = bar.num_boundary + bar.num_points;

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    const double energy =
        bar.bond.energy_coef[b] * health[b] * stretch[b] * stretch[b];

    // Full computational energy, including ghost region.
    energies.total += energy;

    const std::size_t i = bar.bond.i[b];
    const std::size_t j = bar.bond.j[b];

    const bool i_inside = i >= first_inside && i < last_inside;

    const bool j_inside = j >= first_inside && j < last_inside;

    if (i_inside && j_inside) {
      energies.physical += energy;
    } else if (i_inside || j_inside) {
      energies.physical += 0.5 * energy;
    }

    if (bar.bond.can_break[b]) {
      energies.breakable += energy;
    }
  }

  return energies;
}

double
computeMomentum(const std::vector<double>& v, const Bar& bar, double rho) {
  double momentum = 0.0;

  const double point_mass = rho * bar.volume;

  for (const std::size_t i : bar.inside) {
    momentum += point_mass * v[i];
  }

  return momentum;
}

BoundaryReactions computeBoundaryReactions(const std::vector<double>& force,
                                           const Bar& bar) {
  BoundaryReactions reactions{0.0, 0.0};

  for (std::size_t i = 0; i < bar.num_boundary; ++i) {
    reactions.left -= force[i] * bar.volume;
  }

  const std::size_t first_right = bar.num_boundary + bar.num_points;

  for (std::size_t i = first_right; i < bar.num_total; ++i) {
    reactions.right -= force[i] * bar.volume;
  }

  return reactions;
}

// ============================================================================
// Fragment statistics
// ============================================================================

FragmentStatistics computeFragmentStatistics(const std::vector<double>& sizes) {
  FragmentStatistics stats{0.0, 0.0, 0.0};

  if (sizes.empty()) {
    return stats;
  }

  stats.mean_size = std::accumulate(sizes.begin(), sizes.end(), 0.0) /
                    static_cast<double>(sizes.size());

  const auto minmax = std::minmax_element(sizes.begin(), sizes.end());

  stats.min_size = *minmax.first;
  stats.max_size = *minmax.second;

  return stats;
}

// ============================================================================
// Point-field statistics
// ============================================================================

FieldStatistics computeFieldStatistics(const PointFields& fields) {
  FieldStatistics stats{};

  double stress_sum = 0.0;
  double strain_sum = 0.0;
  double damage_sum = 0.0;

  std::size_t stress_count = 0;
  std::size_t strain_count = 0;

  stats.max_stress = std::numeric_limits<double>::lowest();
  stats.max_strain = std::numeric_limits<double>::lowest();

  for (std::size_t p = 0; p < fields.damage.size(); ++p) {
    if (std::isfinite(fields.stress[p])) {
      stress_sum += fields.stress[p];
      ++stress_count;

      stats.max_stress = std::max(stats.max_stress, fields.stress[p]);
    }

    if (std::isfinite(fields.strain[p])) {
      strain_sum += fields.strain[p];
      ++strain_count;

      stats.max_strain = std::max(stats.max_strain, fields.strain[p]);
    }

    damage_sum += fields.damage[p];

    stats.max_damage = std::max(stats.max_damage, fields.damage[p]);
  }

  if (stress_count > 0) {
    stats.average_stress = stress_sum / static_cast<double>(stress_count);

  } else {
    stats.average_stress = 0.0;
    stats.max_stress = 0.0;
  }

  if (strain_count > 0) {
    stats.average_strain = strain_sum / static_cast<double>(strain_count);

  } else {
    stats.average_strain = 0.0;
    stats.max_strain = 0.0;
  }

  if (!fields.damage.empty()) {
    stats.mean_damage = damage_sum / static_cast<double>(fields.damage.size());
  }

  return stats;
}

// ============================================================================
// Bond statistics
// ============================================================================

BondStatistics computeBondStatistics(const std::vector<double>& health,
                                     const std::vector<double>& stretch,
                                     const std::vector<double>& max_stretch,
                                     const Bar& bar) {
  BondStatistics stats{};

  stats.min_health = 1.0;

  std::size_t num_breakable = 0;

  const double eps = 1e-12;

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    if (!bar.bond.can_break[b]) {
      continue;
    }

    ++num_breakable;

    const double h = health[b];

    if (h <= eps) {
      ++stats.num_broken;

    } else if (h >= 1.0 - eps) {
      ++stats.num_intact;

    } else {
      ++stats.num_softening;
    }

    stats.mean_health += h;

    stats.min_health = std::min(stats.min_health, h);

    stats.max_current_stretch = std::max(stats.max_current_stretch, stretch[b]);

    stats.max_historical_stretch =
        std::max(stats.max_historical_stretch, max_stretch[b]);
  }

  if (num_breakable > 0) {
    stats.mean_health /= static_cast<double>(num_breakable);

    stats.fraction_broken = static_cast<double>(stats.num_broken) /
                            static_cast<double>(num_breakable);

  } else {
    stats.min_health = 0.0;
  }

  return stats;
}

// ============================================================================
// Gap connectivity
// ============================================================================

std::vector<double> computeHealthConnectivity(const std::vector<double>& health,
                                              const Bar& bar) {
  std::vector<double> change(bar.num_points, 0.0);

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    if (!bar.bond.can_break[b]) {
      continue;
    }

    const std::size_t i = bar.bond.i[b] - bar.num_boundary;

    const std::size_t j = bar.bond.j[b] - bar.num_boundary;

    change[i] += health[b];
    change[j] -= health[b];
  }

  std::vector<double> connectivity;

  if (bar.num_points > 1) {
    connectivity.resize(bar.num_points - 1, 0.0);
  }

  double current = 0.0;

  for (std::size_t gap = 0; gap + 1 < bar.num_points; ++gap) {
    current += change[gap];
    connectivity[gap] = current;
  }

  return connectivity;
}

// ============================================================================
// Point snapshot
// ============================================================================

void writePointSnapshot(const Parameters& params,
                        const Bar& bar,
                        std::size_t step,
                        double time,
                        const std::vector<double>& u,
                        const std::vector<double>& v,
                        const std::vector<double>& acceleration,
                        const PointFields& fields) {
  const std::string path =
      params.output_dir + "/fields/points_" + std::to_string(step) + ".out";

  std::ofstream output(path);

  if (!output.is_open()) {
    throw std::runtime_error("Could not open point-field file: " + path);
  }

  output << std::setprecision(16);

  output << "# step time point_id x current_x displacement "
         << "velocity acceleration damage strain stress\n";

  for (std::size_t p = 0; p < bar.num_points; ++p) {
    const std::size_t i = bar.inside[p];

    output << step << " " << time << " " << p << " " << bar.x[i] << " "
           << bar.x[i] + u[i] << " " << u[i] << " " << v[i] << " "
           << acceleration[p] << " " << fields.damage[p] << " "
           << fields.strain[p] << " " << fields.stress[p] << "\n";
  }
}

// ============================================================================
// Filtered fragments
// ============================================================================

void writeFilteredFragments(std::ofstream& output,
                            std::size_t step,
                            double time,
                            const FragmentData& fragments,
                            const std::vector<double>& v,
                            const Bar& bar,
                            double rho) {
  const double point_mass = rho * bar.volume;

  const double domain_left = -0.5 * bar.L;

  const double domain_right = 0.5 * bar.L;

  for (std::size_t f = 0; f < fragments.sizes.size(); ++f) {
    const double x_left = (f == 0) ? domain_left : fragments.cut_x[f - 1];

    const double x_right =
        (f < fragments.cut_x.size()) ? fragments.cut_x[f] : domain_right;

    double mass = 0.0;
    double momentum = 0.0;

    for (std::size_t p = 0; p < bar.num_points; ++p) {
      const std::size_t i = bar.inside[p];

      const double x_point = bar.x[i];

      const bool inside_fragment =
          (x_point >= x_left) &&
          ((f + 1 == fragments.sizes.size()) ? x_point <= x_right
                                             : x_point < x_right);

      if (inside_fragment) {
        mass += point_mass;

        momentum += point_mass * v[i];
      }
    }

    const double mean_velocity = (mass > 0.0) ? momentum / mass : 0.0;

    output << step << " " << time << " " << f << " " << x_left << " " << x_right
           << " " << 0.5 * (x_left + x_right) << " " << fragments.sizes[f]
           << " " << mass << " " << mean_velocity << "\n";
  }
}

// ============================================================================
// Raw fragments
// ============================================================================

void writeRawFragments(std::ofstream& output,
                       std::size_t step,
                       double time,
                       const FragmentData& fragments,
                       const Bar& bar) {
  const double domain_left = -0.5 * bar.L;

  const double domain_right = 0.5 * bar.L;

  for (std::size_t f = 0; f < fragments.raw_sizes.size(); ++f) {
    const double x_left = (f == 0) ? domain_left : fragments.raw_cut_x[f - 1];

    const double x_right = (f < fragments.raw_cut_x.size())
                               ? fragments.raw_cut_x[f]
                               : domain_right;

    output << step << " " << time << " " << f << " " << x_left << " " << x_right
           << " " << 0.5 * (x_left + x_right) << " " << fragments.raw_sizes[f]
           << "\n";
  }
}

// ============================================================================
// Connectivity snapshot
// ============================================================================

void writeConnectivitySnapshot(
    const Parameters& params,
    std::size_t step,
    double time,
    const FragmentData& fragments,
    const std::vector<double>& health_connectivity,
    const std::vector<long>& initial_active_crossing_bonds,
    const std::vector<double>& initial_health_connectivity) {
  const std::string path = params.output_dir + "/connectivity/connectivity_" +
                           std::to_string(step) + ".out";

  std::ofstream output(path);

  if (!output.is_open()) {
    throw std::runtime_error("Could not open connectivity file: " + path);
  }

  output << std::setprecision(16);

  output << "# step time gap_id gap_x active_crossing_bonds "
         << "active_connectivity_normalised health_connectivity "
         << "health_connectivity_normalised raw_cut\n";

  for (std::size_t g = 0; g < fragments.gap_x.size(); ++g) {
    double active_normalised = 0.0;
    double health_normalised = 0.0;

    if (initial_active_crossing_bonds[g] > 0) {
      active_normalised =
          static_cast<double>(fragments.active_crossing_bonds[g]) /
          static_cast<double>(initial_active_crossing_bonds[g]);
    }

    if (initial_health_connectivity[g] > 0.0) {
      health_normalised =
          health_connectivity[g] / initial_health_connectivity[g];
    }

    const int raw_cut = fragments.active_crossing_bonds[g] == 0 ? 1 : 0;

    output << step << " " << time << " " << g << " " << fragments.gap_x[g]
           << " " << fragments.active_crossing_bonds[g] << " "
           << active_normalised << " " << health_connectivity[g] << " "
           << health_normalised << " " << raw_cut << "\n";
  }
}

} // namespace

// ============================================================================
// Initialise persistent output files
// ============================================================================

PDOutputFiles initialisePDOutputs(const Parameters& params) {
  std::filesystem::create_directories(params.output_dir);

  std::filesystem::create_directories(params.output_dir + "/fields");

  std::filesystem::create_directories(params.output_dir + "/connectivity");

  PDOutputFiles files{};

  const std::string diagnostics_path =
      params.output_dir + "/" + params.diagnostics_file;

  const std::string fragments_path =
      params.output_dir + "/" + params.fragment_file;

  const std::string raw_fragments_path =
      params.output_dir + "/raw_fragments.out";

  const std::string pd_diagnostics_path =
      params.output_dir + "/pd_diagnostics.out";

  const std::string bond_events_path = params.output_dir + "/bond_events.out";

  files.diagnostics = new std::ofstream(diagnostics_path);

  files.fragments = new std::ofstream(fragments_path);

  files.raw_fragments = new std::ofstream(raw_fragments_path);

  files.pd_diagnostics = new std::ofstream(pd_diagnostics_path);

  files.bond_events = new std::ofstream(bond_events_path);

  if (!files.diagnostics->is_open() || !files.fragments->is_open() ||
      !files.raw_fragments->is_open() || !files.pd_diagnostics->is_open() ||
      !files.bond_events->is_open()) {
    closePDOutputs(files);

    throw std::runtime_error(
        "Could not initialise one or more PD output files.");
  }

  *files.diagnostics << std::setprecision(16);
  *files.fragments << std::setprecision(16);
  *files.raw_fragments << std::setprecision(16);
  *files.pd_diagnostics << std::setprecision(16);
  *files.bond_events << std::setprecision(16);

  *files.diagnostics << "# step time dt displacement strain "
                     << "reaction_left reaction_right "
                     << "kinetic_energy elastic_energy "
                     << "reversible_cohesive_energy fracture_energy "
                     << "contact_energy external_work total_energy "
                     << "energy_error num_fragments "
                     << "mean_fragment_size min_fragment_size "
                     << "max_fragment_size momentum "
                     << "average_stress max_stress "
                     << "average_strain max_strain "
                     << "simulation_wall_time "
                     << "simulation_wall_time_since_last_dump\n";

  *files.fragments << "# step time fragment_id "
                   << "x_left x_right centre size "
                   << "mass mean_velocity\n";

  *files.raw_fragments << "# step time fragment_id "
                       << "x_left x_right centre size\n";

  *files.pd_diagnostics << "# step time elastic_energy_breakable "
                        << "raw_num_fragments "
                        << "num_intact_bonds "
                        << "num_softening_bonds "
                        << "num_broken_bonds "
                        << "fraction_broken "
                        << "new_softening_bonds "
                        << "new_broken_bonds "
                        << "mean_health "
                        << "min_health "
                        << "mean_damage "
                        << "max_damage "
                        << "max_current_stretch "
                        << "max_historical_stretch\n";

  *files.bond_events << "# step time bond_id i j "
                     << "x_i x_j x_mid length event "
                     << "stretch max_stretch health "
                     << "s_elastic s_break dissipated_increment\n";

  return files;
}

// ============================================================================
// Close persistent output files
// ============================================================================

void closePDOutputs(PDOutputFiles& files) {
  if (files.diagnostics != nullptr) {
    files.diagnostics->close();
    delete files.diagnostics;
    files.diagnostics = nullptr;
  }

  if (files.fragments != nullptr) {
    files.fragments->close();
    delete files.fragments;
    files.fragments = nullptr;
  }

  if (files.raw_fragments != nullptr) {
    files.raw_fragments->close();
    delete files.raw_fragments;
    files.raw_fragments = nullptr;
  }

  if (files.pd_diagnostics != nullptr) {
    files.pd_diagnostics->close();
    delete files.pd_diagnostics;
    files.pd_diagnostics = nullptr;
  }

  if (files.bond_events != nullptr) {
    files.bond_events->close();
    delete files.bond_events;
    files.bond_events = nullptr;
  }
}

// ============================================================================
// Static PD metadata
// ============================================================================

void writePDMetadata(const Parameters& params, const Bar& bar) {
  const std::string path = params.output_dir + "/pd_metadata.out";

  std::ofstream output(path);

  if (!output.is_open()) {
    throw std::runtime_error("Could not open metadata file: " + path);
  }

  output << std::setprecision(16);

  std::size_t num_breakable = 0;

  double s_elastic_sum = 0.0;
  double s_break_sum = 0.0;

  double s_elastic_min = std::numeric_limits<double>::max();

  double s_break_min = std::numeric_limits<double>::max();

  double s_elastic_max = 0.0;
  double s_break_max = 0.0;

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    if (!bar.bond.can_break[b]) {
      continue;
    }

    ++num_breakable;

    const double s_elastic = bar.bond.s_elastic[b];

    const double s_break = bar.bond.s_break[b];

    s_elastic_sum += s_elastic;
    s_break_sum += s_break;

    s_elastic_min = std::min(s_elastic_min, s_elastic);

    s_elastic_max = std::max(s_elastic_max, s_elastic);

    s_break_min = std::min(s_break_min, s_break);

    s_break_max = std::max(s_break_max, s_break);
  }

  output << "length " << bar.L << "\n";
  output << "area " << bar.area << "\n";

  output << "density " << params.density << "\n";

  output << "E " << params.E << "\n";

  output << "strength " << params.strength << "\n";

  output << "toughness " << params.toughness << "\n";

  output << "strain_rate " << params.strain_rate << "\n";

  output << "dx " << bar.dx << "\n";

  output << "horizon " << bar.horizon << "\n";

  output << "m " << bar.m << "\n";

  output << "wave_speed " << bar.wave_speed << "\n";

  output << "dt " << bar.dt << "\n";

  output << "time_step_factor " << params.time_step_factor << "\n";

  output << "num_points " << bar.num_points << "\n";

  output << "num_boundary " << bar.num_boundary << "\n";

  output << "num_total " << bar.num_total << "\n";

  output << "num_bonds " << bar.bond.count << "\n";

  output << "num_breakable_bonds " << num_breakable << "\n";

  output << "bond_law " << params.bond_law << "\n";

  output << "bond_shape " << params.bond_shape << "\n";

  output << "max_time " << params.max_time << "\n";

  output << "enable_early_stopping " << params.enable_early_stopping << "\n";

  output << "dump_freq " << params.dump_freq << "\n";

  output << "check_frequency " << params.check_frequency << "\n";

  output << "stable_steps " << params.stable_steps << "\n";

  // The current default countFragments() filtering length.
  output << "fragment_filter_length " << bar.horizon << "\n";

  output << "fragment_filter_rule horizon\n";

  if (num_breakable > 0) {
    output << "mean_s_elastic_breakable "
           << s_elastic_sum / static_cast<double>(num_breakable) << "\n";

    output << "min_s_elastic_breakable " << s_elastic_min << "\n";

    output << "max_s_elastic_breakable " << s_elastic_max << "\n";

    output << "mean_s_break_breakable "
           << s_break_sum / static_cast<double>(num_breakable) << "\n";

    output << "min_s_break_breakable " << s_break_min << "\n";

    output << "max_s_break_breakable " << s_break_max << "\n";
  }
}

// ============================================================================
// Static spatial material properties
// ============================================================================

void writePDMaterialProperties(const Parameters& params, const Bar& bar) {
  const std::string path = params.output_dir + "/material_properties.out";

  std::ofstream output(path);

  if (!output.is_open()) {
    throw std::runtime_error("Could not open material-properties file: " +
                             path);
  }

  output << std::setprecision(16);

  output << "# point_id global_id x strength toughness\n";

  for (std::size_t p = 0; p < bar.num_points; ++p) {
    const std::size_t i = bar.inside[p];

    output << p << " " << i << " " << bar.x[i] << " " << bar.strength_pt[i]
           << " " << bar.toughness_pt[i] << "\n";
  }
}

// ============================================================================
// Bond-event output
// ============================================================================

void writePDBondEvents(std::ofstream& output,
                       const std::vector<BondEvent>& events) {
  for (const BondEvent& event : events) {
    output << event.step << " " << event.time << " " << event.bond_id << " "
           << event.i << " " << event.j << " " << event.x_i << " " << event.x_j
           << " " << event.x_mid << " " << event.length << " " << event.event
           << " " << event.stretch << " " << event.max_stretch << " "
           << event.health << " " << event.s_elastic << " " << event.s_break
           << " " << event.dissipated_increment << "\n";
  }
}

// ============================================================================
// Complete simulation dump
// ============================================================================

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
                      PDOutputFiles& files) {
  if (files.diagnostics == nullptr || files.fragments == nullptr ||
      files.raw_fragments == nullptr || files.pd_diagnostics == nullptr) {
    throw std::runtime_error("PD output files are not initialised.");
  }

  const FragmentData fragments = countFragments(bar, health);

  const PointFields fields = computePointFields(u, health, stretch, bar);

  const BoundaryReactions reactions = computeBoundaryReactions(force, bar);

  const FragmentStatistics fragment_stats =
      computeFragmentStatistics(fragments.sizes);

  const FieldStatistics field_stats = computeFieldStatistics(fields);

  const BondStatistics bond_stats =
      computeBondStatistics(health, stretch, max_stretch, bar);

  const std::vector<double> health_connectivity =
      computeHealthConnectivity(health, bar);

  const double kinetic = computeKineticEnergy(v, bar, params.density);

  const ElasticEnergies elastic = computeElasticEnergies(health, stretch, bar);

  const double total_energy = kinetic + elastic.total + dissipated;

  const double energy_error =
      external_work - (total_energy - initial_total_energy);

  const double error_scale =
      std::max({std::abs(total_energy),
                std::abs(initial_total_energy + external_work),
                std::numeric_limits<double>::epsilon()});

  const double relative_energy_error = std::abs(energy_error) / error_scale;

  const double momentum = computeMomentum(v, bar, params.density);

  const double displacement = rate * bar.L * time;

  const double strain = rate * time;

  // ------------------------------------------------------------------------
  // Common CZM-compatible diagnostics.

  *files.diagnostics << step << " " << time << " " << bar.dt << " "
                     << displacement << " " << strain << " " << reactions.left
                     << " " << reactions.right << " " << kinetic << " "
                     << elastic.physical << " " << 0.0 << " " << dissipated
                     << " " << 0.0 << " " << external_work << " "
                     << total_energy << " " << energy_error << " "
                     << fragments.num_fragments << " "
                     << fragment_stats.mean_size << " "
                     << fragment_stats.min_size << " "
                     << fragment_stats.max_size << " " << momentum << " "
                     << field_stats.average_stress << " "
                     << field_stats.max_stress << " "
                     << field_stats.average_strain << " "
                     << field_stats.max_strain << " " << simulation_wall_time
                     << " " << simulation_wall_time_since_last_dump << "\n";

  // ------------------------------------------------------------------------
  // Filtered fragments.

  writeFilteredFragments(
      *files.fragments, step, time, fragments, v, bar, params.density);

  // ------------------------------------------------------------------------
  // Raw topology fragments.
  //
  // Mass and mean velocity are deliberately not repeated here. The raw
  // fragments are retained to inspect the connectivity topology before the
  // horizon-based fragment filter is applied.

  writeRawFragments(*files.raw_fragments, step, time, fragments, bar);

  // ------------------------------------------------------------------------
  // PD-specific evolving diagnostics.

  *files.pd_diagnostics << step << " " << time << " " << elastic.breakable
                        << " " << fragments.raw_num_fragments << " "
                        << bond_stats.num_intact << " "
                        << bond_stats.num_softening << " "
                        << bond_stats.num_broken << " "
                        << bond_stats.fraction_broken << " "
                        << new_softening_since_last_dump << " "
                        << new_broken_since_last_dump << " "
                        << bond_stats.mean_health << " "
                        << bond_stats.min_health << " "
                        << field_stats.mean_damage << " "
                        << field_stats.max_damage << " "
                        << bond_stats.max_current_stretch << " "
                        << bond_stats.max_historical_stretch << "\n";

  // ------------------------------------------------------------------------
  // Spatial fields.

  writePointSnapshot(params, bar, step, time, u, v, acceleration, fields);

  writeConnectivitySnapshot(params,
                            step,
                            time,
                            fragments,
                            health_connectivity,
                            initial_active_crossing_bonds,
                            initial_health_connectivity);

  return relative_energy_error;
}

// ============================================================================
// Final damaged-bond state
// ============================================================================

void writePDFinalDamagedBonds(const Parameters& params,
                              const Bar& bar,
                              const std::vector<double>& u,
                              const std::vector<double>& health,
                              const std::vector<double>& stretch,
                              const std::vector<double>& max_stretch,
                              const std::vector<double>& bond_dissipated) {
  const std::string path = params.output_dir + "/damaged_bonds_final.out";

  std::ofstream output(path);

  if (!output.is_open()) {
    throw std::runtime_error("Could not open final damaged-bond file: " + path);
  }

  output << std::setprecision(16);

  output << "# bond_id i j x_i x_j x_mid "
         << "current_x_i current_x_j current_x_mid "
         << "length final_stretch health max_stretch "
         << "s_elastic s_break dissipated\n";

  const double eps = 1e-12;

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    if (!bar.bond.can_break[b]) {
      continue;
    }

    if (health[b] >= 1.0 - eps && bond_dissipated[b] <= 0.0) {
      continue;
    }

    const std::size_t i = bar.bond.i[b];

    const std::size_t j = bar.bond.j[b];

    const double x_i = bar.x[i];

    const double x_j = bar.x[j];

    const double current_x_i = x_i + u[i];

    const double current_x_j = x_j + u[j];

    output << b << " " << i - bar.num_boundary << " " << j - bar.num_boundary
           << " " << x_i << " " << x_j << " " << 0.5 * (x_i + x_j) << " "
           << current_x_i << " " << current_x_j << " "
           << 0.5 * (current_x_i + current_x_j) << " " << bar.bond.length[b]
           << " " << stretch[b] << " " << health[b] << " " << max_stretch[b]
           << " " << bar.bond.s_elastic[b] << " " << bar.bond.s_break[b] << " "
           << bond_dissipated[b] << "\n";
  }
}
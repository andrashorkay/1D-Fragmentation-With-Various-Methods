#include "simulate_bar.hh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bond_health.hh"
#include "count_fragments.hh"
#include "parameters.hh"
#include "pd_diagnostic_writer.hh"

namespace {

struct StretchData {
  std::vector<double> stretch;
  std::vector<double> direction;
};

// ============================================================================
// Bond kinematics
// ============================================================================

StretchData bondStretch(const std::vector<double>& u,
                        const std::vector<double>& x,
                        const std::vector<std::size_t>& bi,
                        const std::vector<std::size_t>& bj,
                        const std::vector<double>& L0) {
  const std::size_t num_bonds = bi.size();

  StretchData data;

  data.stretch.resize(num_bonds);
  data.direction.resize(num_bonds);

  for (std::size_t b = 0; b < num_bonds; ++b) {
    const double yi = x[bi[b]] + u[bi[b]];
    const double yj = x[bj[b]] + u[bj[b]];

    const double d = yj - yi;

    data.stretch[b] = (std::abs(d) - L0[b]) / L0[b];

    if (d > 0.0) {
      data.direction[b] = 1.0;

    } else if (d < 0.0) {
      data.direction[b] = -1.0;

    } else {
      data.direction[b] = 0.0;
    }
  }

  return data;
}

// ============================================================================
// Peridynamic forces
// ============================================================================

std::vector<double> bondForces(const std::vector<double>& health,
                               const std::vector<double>& k_bond,
                               const std::vector<double>& stretch,
                               const std::vector<double>& direction,
                               const std::vector<std::size_t>& bi,
                               const std::vector<std::size_t>& bj,
                               std::size_t n_all) {
  std::vector<double> force(n_all, 0.0);

  for (std::size_t b = 0; b < health.size(); ++b) {
    const double fb = health[b] * k_bond[b] * stretch[b] * direction[b];

    force[bi[b]] += fb;
    force[bj[b]] -= fb;
  }

  return force;
}

// ========================================================================== //
// Prescribed boundary motion

void moveBoundary(std::vector<double>& u,
                  std::vector<double>& v,
                  double time,
                  double rate,
                  const std::vector<double>& x,
                  const std::vector<std::size_t>& boundary) {
  for (const std::size_t i : boundary) {
    u[i] = rate * time * x[i];

    v[i] = rate * x[i];
  }
}

// ========================================================================== //
// Small summary helpers

double
computeKineticEnergy(const std::vector<double>& v, const Bar& bar, double rho) {
  double kinetic = 0.0;

  for (const std::size_t i : bar.inside) {
    kinetic += v[i] * v[i];
  }

  return 0.5 * rho * bar.volume * kinetic;
}

double computePhysicalElasticEnergy(const std::vector<double>& health,
                                    const std::vector<double>& stretch,
                                    const Bar& bar) {
  double elastic = 0.0;

  const std::size_t first_inside = bar.num_boundary;
  const std::size_t last_inside = bar.num_boundary + bar.num_points;

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    const std::size_t i = bar.bond.i[b];
    const std::size_t j = bar.bond.j[b];

    const bool i_inside = i >= first_inside && i < last_inside;

    const bool j_inside = j >= first_inside && j < last_inside;

    double weight = 0.0;

    if (i_inside && j_inside) {
      weight = 1.0;
    } else if (i_inside || j_inside) {
      weight = 0.5;
    }

    elastic +=
        weight * bar.bond.energy_coef[b] * health[b] * stretch[b] * stretch[b];
  }

  return elastic;
}

double computeTotalElasticEnergy(const std::vector<double>& health,
                                 const std::vector<double>& stretch,
                                 const Bar& bar) {
  double elastic = 0.0;

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    elastic += bar.bond.energy_coef[b] * health[b] * stretch[b] * stretch[b];
  }

  return elastic;
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

} // namespace

// ========================================================================== //
// Main simulation

SimulationResults simulateBar(const Parameters& params, const Bar& bar) {
  const auto wall_start = std::chrono::steady_clock::now();

  const double rate = params.strain_rate;

  const std::vector<double>& x = bar.x;
  const std::vector<std::size_t>& inside = bar.inside;
  const std::vector<std::size_t>& boundary = bar.boundary;

  const double rho = params.density;
  const double volume = bar.volume;
  const double dt = bar.dt;

  const std::vector<std::size_t>& bi = bar.bond.i;
  const std::vector<std::size_t>& bj = bar.bond.j;

  const std::vector<double>& L0 = bar.bond.length;
  const std::vector<double>& k_bond = bar.bond.k;

  const std::vector<double>& energy_coef = bar.bond.energy_coef;

  // ======================================================================== //
  // Input checks.

  if (params.max_time <= 0.0) {
    throw std::runtime_error("max_time must be greater than zero.");
  }

  if (params.dump_freq == 0) {
    throw std::runtime_error("dump_freq must be greater than zero.");
  }

  if (params.enable_early_stopping) {
    if (params.check_frequency == 0) {
      throw std::runtime_error("check_frequency must be greater than zero when "
                               "early stopping is enabled.");
    }

    if (params.stable_steps == 0) {
      throw std::runtime_error("stable_steps must be greater than zero when "
                               "early stopping is enabled.");
    }
  }

  if (dt <= 0.0) {
    throw std::runtime_error("Time step must be greater than zero.");
  }

  // Initialise output.
  PDOutputFiles output_files = initialisePDOutputs(params);

  writePDMetadata(params, bar);
  writePDMaterialProperties(params, bar);

  // Identify breakable bonds.
  std::vector<std::size_t> breakable;

  breakable.reserve(bar.bond.count);

  for (std::size_t b = 0; b < bar.bond.count; ++b) {
    if (bar.bond.can_break[b]) {
      breakable.push_back(b);
    }
  }

  std::vector<double> s_elastic(breakable.size());

  std::vector<double> s_break(breakable.size());

  std::vector<double> energy_coef_breakable(breakable.size());

  for (std::size_t q = 0; q < breakable.size(); ++q) {
    const std::size_t b = breakable[q];

    s_elastic[q] = bar.bond.s_elastic[b];

    s_break[q] = bar.bond.s_break[b];

    energy_coef_breakable[q] = energy_coef[b];
  }

  // Fragment-convergence state.
  std::size_t previous_nb_fragments = 0;

  std::size_t stable_fragment_checks = 0;

  bool have_previous_fragment_count = false;

  // Initial displacement and velocity.
  std::vector<double> u(bar.num_total, 0.0);

  std::vector<double> v(bar.num_total);

  for (std::size_t i = 0; i < bar.num_total; ++i) {
    v[i] = rate * x[i];
  }

  // Bond state.
  std::vector<double> health(bar.bond.count, 1.0);

  std::vector<double> health_breakable(breakable.size(), 1.0);

  std::vector<double> max_stretch(bar.bond.count, 0.0);

  std::vector<double> bond_dissipated(bar.bond.count, 0.0);

  // Initial force state.
  moveBoundary(u, v, 0.0, rate, x, boundary);

  StretchData bond_data = bondStretch(u, x, bi, bj, L0);

  std::vector<double> force = bondForces(health,
                                         k_bond,
                                         bond_data.stretch,
                                         bond_data.direction,
                                         bi,
                                         bj,
                                         bar.num_total);

  std::vector<double> acceleration(bar.num_points);

  for (std::size_t p = 0; p < bar.num_points; ++p) {
    acceleration[p] = force[inside[p]] / rho;
  }

  // Energy and external work.
  double dissipated = 0.0;
  double work_in = 0.0;

  double power_prev = 0.0;

  for (const std::size_t i : boundary) {
    power_prev -= force[i] * v[i] * volume;
  }

  const double E_kin_0 = computeKineticEnergy(v, bar, rho);

  const double E_elastic_total_0 =
      computeTotalElasticEnergy(health, bond_data.stretch, bar);

  const double E_tot_0 = E_kin_0 + E_elastic_total_0;

  // Initial connectivity.
  //
  // At t = 0 every breakable bond has health = 1, so the health-weighted
  // connectivity is identical to the number of active crossing bonds.

  const FragmentData initial_fragments = countFragments(bar, health);

  const std::vector<long> initial_active_crossing_bonds =
      initial_fragments.active_crossing_bonds;

  std::vector<double> initial_health_connectivity(
      initial_active_crossing_bonds.size());

  for (std::size_t g = 0; g < initial_active_crossing_bonds.size(); ++g) {
    initial_health_connectivity[g] =
        static_cast<double>(initial_active_crossing_bonds[g]);
  }
  // Time and timing state.

  std::size_t step = 0;
  double time = 0.0;

  double simulation_wall_time = 0.0;

  double simulation_wall_time_at_last_dump = 0.0;

  double max_relative_energy_error = 0.0;

  std::size_t new_softening_since_last_dump = 0;

  std::size_t new_broken_since_last_dump = 0;

  // Initial state output.

  max_relative_energy_error =
      writePDSimulationDump(params,
                            bar,
                            step,
                            time,
                            rate,
                            u,
                            v,
                            acceleration,
                            force,
                            health,
                            bond_data.stretch,
                            max_stretch,
                            dissipated,
                            work_in,
                            E_tot_0,
                            0.0,
                            0.0,
                            0,
                            0,
                            initial_active_crossing_bonds,
                            initial_health_connectivity,
                            output_files);

  std::cout << "Time step: " << dt << " s\n";

  std::cout << "Strain rate: " << rate << " /s\n";

  std::cout << "Maximum time: " << params.max_time << " s\n";

  std::cout << "Early stopping: "
            << (params.enable_early_stopping ? "enabled" : "disabled") << "\n";

  if (params.enable_early_stopping) {
    std::cout << "Fragment check frequency: " << params.check_frequency
              << " steps\n";

    std::cout << "Stable fragment checks required: " << params.stable_steps
              << "\n";
  }

  // Time integration.

  while (time + dt <= params.max_time) {
    const auto step_start = std::chrono::steady_clock::now();

    ++step;

    time = static_cast<double>(step) * dt;
    // 1-2. Half-step velocity and displacement update.

    std::vector<double> v_half(bar.num_points);

    for (std::size_t p = 0; p < bar.num_points; ++p) {
      const std::size_t i = inside[p];

      v_half[p] = v[i] + 0.5 * dt * acceleration[p];

      u[i] += dt * v_half[p];
    }

    // 3. Prescribed boundary motion.
    moveBoundary(u, v, time, rate, x, boundary);

    // 4. Bond stretch and irreversible damage.
    bond_data = bondStretch(u, x, bi, bj, L0);

    std::vector<double> stretch_breakable(breakable.size());

    std::vector<double> max_stretch_breakable(breakable.size());

    for (std::size_t q = 0; q < breakable.size(); ++q) {
      const std::size_t b = breakable[q];

      stretch_breakable[q] = bond_data.stretch[b];

      max_stretch[b] = std::max(max_stretch[b], stretch_breakable[q]);

      max_stretch_breakable[q] = max_stretch[b];
    }

    std::vector<double> new_health =
        bondHealth(params.bond_law, max_stretch_breakable, s_elastic, s_break);

    std::vector<BondEvent> step_events;

    const double health_eps = 1e-12;

    for (std::size_t q = 0; q < breakable.size(); ++q) {
      const std::size_t b = breakable[q];

      const double old_health = health_breakable[q];

      new_health[q] = std::min(old_health, new_health[q]);

      const double health_loss = old_health - new_health[q];

      const double dissipated_increment = energy_coef_breakable[q] *
                                          health_loss * stretch_breakable[q] *
                                          stretch_breakable[q];

      dissipated += dissipated_increment;

      bond_dissipated[b] += dissipated_increment;

      // Bond events.
      auto addEvent = [&](const std::string& event_name) {
        BondEvent event{};

        event.step = step;
        event.time = time;

        event.bond_id = b;

        event.i = bi[b] - bar.num_boundary;

        event.j = bj[b] - bar.num_boundary;

        event.x_i = x[bi[b]];

        event.x_j = x[bj[b]];

        event.x_mid = 0.5 * (event.x_i + event.x_j);

        event.length = L0[b];

        event.event = event_name;

        event.stretch = stretch_breakable[q];

        event.max_stretch = max_stretch[b];

        event.health = new_health[q];

        event.s_elastic = s_elastic[q];

        event.s_break = s_break[q];

        event.dissipated_increment = dissipated_increment;

        step_events.push_back(event);
      };

      if (old_health >= 1.0 - health_eps && new_health[q] < 1.0 - health_eps) {
        addEvent("softening_started");

        ++new_softening_since_last_dump;
      }

      if (old_health > health_eps && new_health[q] <= health_eps) {
        addEvent("fully_broken");

        ++new_broken_since_last_dump;
      }

      health_breakable[q] = new_health[q];

      health[b] = new_health[q];
    }

    // 5-6. New forces and final velocity update.

    force = bondForces(health,
                       k_bond,
                       bond_data.stretch,
                       bond_data.direction,
                       bi,
                       bj,
                       bar.num_total);

    for (std::size_t p = 0; p < bar.num_points; ++p) {
      const std::size_t i = inside[p];

      acceleration[p] = force[i] / rho;

      v[i] = v_half[p] + 0.5 * dt * acceleration[p];
    }

    // External work.

    double power_now = 0.0;

    for (const std::size_t i : boundary) {
      power_now -= force[i] * v[i] * volume;
    }

    work_in += 0.5 * dt * (power_prev + power_now);

    power_prev = power_now;

    // Fragment-count convergence.

    if (params.enable_early_stopping && step % params.check_frequency == 0) {
      const FragmentData fragment_check = countFragments(bar, health);

      const std::size_t nb_fragments = fragment_check.num_fragments;

      if (have_previous_fragment_count &&
          nb_fragments == previous_nb_fragments && nb_fragments > 1) {
        ++stable_fragment_checks;

      } else {
        stable_fragment_checks = 0;
      }

      previous_nb_fragments = nb_fragments;

      have_previous_fragment_count = true;

      std::cout << "Step: " << step << " -- fragments = " << nb_fragments
                << " -- stable checks = " << stable_fragment_checks << "/"
                << params.stable_steps << "\n";
    }

    // ------------------------------------------------------------------------
    // Stop compute-only timer before output.

    const auto step_end = std::chrono::steady_clock::now();

    simulation_wall_time +=
        std::chrono::duration<double>(step_end - step_start).count();

    // ------------------------------------------------------------------------
    // Bond-event output.

    if (!step_events.empty()) {
      writePDBondEvents(*output_files.bond_events, step_events);
    }

    // ------------------------------------------------------------------------
    // Periodic diagnostic output.

    if (step % params.dump_freq == 0) {
      const double simulation_wall_time_since_last_dump =
          simulation_wall_time - simulation_wall_time_at_last_dump;

      const double relative_energy_error =
          writePDSimulationDump(params,
                                bar,
                                step,
                                time,
                                rate,
                                u,
                                v,
                                acceleration,
                                force,
                                health,
                                bond_data.stretch,
                                max_stretch,
                                dissipated,
                                work_in,
                                E_tot_0,
                                simulation_wall_time,
                                simulation_wall_time_since_last_dump,
                                new_softening_since_last_dump,
                                new_broken_since_last_dump,
                                initial_active_crossing_bonds,
                                initial_health_connectivity,
                                output_files);

      max_relative_energy_error =
          std::max(max_relative_energy_error, relative_energy_error);

      simulation_wall_time_at_last_dump = simulation_wall_time;

      new_softening_since_last_dump = 0;
      new_broken_since_last_dump = 0;
    }

    // Optional early termination after all timestep output has been written.
    if (params.enable_early_stopping &&
        stable_fragment_checks >= params.stable_steps) {
      break;
    }
  }

  // ==========================================================================
  // Final state output.

  if (step > 0 && step % params.dump_freq != 0) {
    const double simulation_wall_time_since_last_dump =
        simulation_wall_time - simulation_wall_time_at_last_dump;

    const double relative_energy_error =
        writePDSimulationDump(params,
                              bar,
                              step,
                              time,
                              rate,
                              u,
                              v,
                              acceleration,
                              force,
                              health,
                              bond_data.stretch,
                              max_stretch,
                              dissipated,
                              work_in,
                              E_tot_0,
                              simulation_wall_time,
                              simulation_wall_time_since_last_dump,
                              new_softening_since_last_dump,
                              new_broken_since_last_dump,
                              initial_active_crossing_bonds,
                              initial_health_connectivity,
                              output_files);

    max_relative_energy_error =
        std::max(max_relative_energy_error, relative_energy_error);
  }

  // --------------------------------------------------------------------------
  // Stopping reason.

  std::string stopping_reason;

  if (params.enable_early_stopping &&
      stable_fragment_checks >= params.stable_steps) {
    stopping_reason = "fragment_count_converged";

  } else {
    stopping_reason = "maximum_time_reached";
  }

  // --------------------------------------------------------------------------
  // Final fragments.

  const FragmentData final_fragments = countFragments(bar, health);

  // --------------------------------------------------------------------------
  // Final energy state.

  const double E_kin_final = computeKineticEnergy(v, bar, rho);

  const double E_elastic_total_final =
      computeTotalElasticEnergy(health, bond_data.stretch, bar);

  const double E_elastic_physical_final =
      computePhysicalElasticEnergy(health, bond_data.stretch, bar);

  const double E_tot_final = E_kin_final + E_elastic_total_final + dissipated;

  const double final_energy_error = work_in - (E_tot_final - E_tot_0);

  const double final_momentum = computeMomentum(v, bar, rho);

  // --------------------------------------------------------------------------
  // Final damaged-bond state.

  writePDFinalDamagedBonds(
      params, bar, u, health, bond_data.stretch, max_stretch, bond_dissipated);

  // Flush and close all diagnostic output before measuring total wall time.

  closePDOutputs(output_files);

  const auto wall_end = std::chrono::steady_clock::now();

  const double wall_time =
      std::chrono::duration<double>(wall_end - wall_start).count();

  // --------------------------------------------------------------------------
  // Final result structure.

  SimulationResults results{};

  results.steps = step;
  results.physical_time = time;

  results.stopping_reason = stopping_reason;

  results.wall_time = wall_time;

  results.simulation_wall_time = simulation_wall_time;

  results.kinetic_energy = E_kin_final;

  results.elastic_energy = E_elastic_physical_final;

  results.fracture_energy = dissipated;

  results.external_work = work_in;

  results.total_energy = E_tot_final;

  results.energy_error = final_energy_error;

  results.max_relative_energy_error = max_relative_energy_error;

  results.momentum = final_momentum;

  results.num_fragments = final_fragments.num_fragments;

  results.cuts = final_fragments.cut_x;

  results.sizes = final_fragments.sizes;

  results.raw_num_fragments = final_fragments.raw_num_fragments;

  results.raw_cuts = final_fragments.raw_cut_x;

  results.raw_sizes = final_fragments.raw_sizes;

  results.bond_health = std::move(health);

  results.bond_max_stretch = std::move(max_stretch);

  results.bond_dissipated = std::move(bond_dissipated);

  // --------------------------------------------------------------------------
  // Lightweight final summary.

  const std::string summary_path = params.output_dir + "/summary.out";

  std::ofstream summary_output(summary_path);

  if (!summary_output.is_open()) {
    throw std::runtime_error("Could not open summary file: " + summary_path);
  }

  summary_output << std::setprecision(16);

  summary_output << "# physical_time wall_time simulation_wall_time "
                 << "steps num_fragments raw_num_fragments "
                 << "energy_error max_relative_energy_error "
                 << "stopping_reason\n";

  summary_output << results.physical_time << " " << results.wall_time << " "
                 << results.simulation_wall_time << " " << results.steps << " "
                 << results.num_fragments << " " << results.raw_num_fragments
                 << " " << results.energy_error << " "
                 << results.max_relative_energy_error << " "
                 << results.stopping_reason << "\n";

  // --------------------------------------------------------------------------
  // Terminal summary.

  std::cout << "Done in " << results.wall_time << " s. "
            << results.num_fragments << " fragments.\n";

  std::cout << "Final physical time: " << results.physical_time << " s\n";

  std::cout << "Final step: " << results.steps << "\n";

  std::cout << "Stopping reason: " << results.stopping_reason << "\n\n";

  return results;
}
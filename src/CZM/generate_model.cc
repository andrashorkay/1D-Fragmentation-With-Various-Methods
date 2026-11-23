#include "generate_model.hh"
#include "material_field.hh"
#include <aka_common.hh>
#include <cmath>
#include <iostream>
#include <mesh.hh>
#include <non_linear_solver.hh>
#include <solid_mechanics_model_cohesive.hh>
#include <stdexcept>
using namespace akantu;

/// @brief Applies the initial velocity field as \(v(x,0)=\dot{\varepsilon}x\)
/// to the nodes of the mesh.
/// @param model corresponding to the cohesive zone model
/// @param strain_rate the applied strain rate \(\dot{\varepsilon}\)
void applyInitialVelocity(SolidMechanicsModelCohesive& model,
                          Real strain_rate) {
  auto& mesh = model.getMesh();
  auto& nodes = mesh.getNodes();
  auto& velocities = model.getVelocity();

  for (Int i = 0; i < nodes.size(); ++i) {
    const Real x = nodes(i, 0);
    velocities(i, 0) = strain_rate * x;
    velocities(i, 1) = 0.0;
  }
}

/// @brief Only vertical facets can nucleate cohesive elements. This allows the
/// 2D bar to break as if it were a 1D bar, so only vertical cracks can form.
/// @param model corresponding to the cohesive zone model
/// @param mesh the mesh of the 2D bar (pseudo-1D bar)
void restrictCohesiveInsertion(SolidMechanicsModelCohesive& model, Mesh& mesh) {
  const Int spatial_dimension = mesh.getSpatialDimension();
  auto& mesh_facets = mesh.getMeshFacets();
  const auto& nodes = mesh.getNodes();

  auto& inserter = model.getElementInserter();

  // Loop over all 1D facet types
  for (auto facet_type : mesh_facets.elementTypes(spatial_dimension - 1)) {
    const auto& connectivity = mesh_facets.getConnectivity(facet_type);
    auto& check = inserter.getCheckFacets(facet_type, _not_ghost);

    // Loop over every facet of this type
    for (Int el = 0; el < connectivity.size(); ++el) {
      const Int node_0 = connectivity(el, 0);
      const Int node_1 = connectivity(el, 1);

      const Real dx = nodes(node_1, 0) - nodes(node_0, 0);
      const Real dy = nodes(node_1, 1) - nodes(node_0, 1);
      const Real norm = std::sqrt(dx * dx + dy * dy);

      // Dot product with the vertical unit vector (0,1)
      const Real direction = dy / norm;

      // Only vertical facets are allowed to nucleate cohesive elements
      if (std::abs(direction) < 0.99) {
        check(el) = false;
      }
    }
  }
}

/// @brief Initialises the extrinsic cohesive-zone model and solvers.
/// @param model Cohesive zone model to be initialised
/// @param mesh Mesh of the 2D bar (pseudo-1D bar)
/// @param strain_rate strain rate applied to the bar
/// @param material_field material field containing the spatial varying
/// properties of the bar.
void initialiseModel(SolidMechanicsModelCohesive& model,
                     Mesh& mesh,
                     Real strain_rate,
                     const MaterialField& material_field) {
  model.initFull(_analysis_method = _static, _is_extrinsic = true);

  // Configure the static solver.
  auto& solver = model.getNonLinearSolver("static");

  solver.set("max_iterations", 100);
  solver.set("threshold", 1e-10);
  solver.set("convergence_type", SolveConvergenceCriteria::_residual);

  // Add explicit lumped-mass solver for the dynamic simulation.
  model.initNewSolver(_explicit_lumped_mass);

  // Restrict cohesive insertion.
  const bool use_automatic_insertion = false;

  if (use_automatic_insertion) {
    model.updateAutomaticInsertion();
  } else {
    restrictCohesiveInsertion(model, mesh);
  }

  // Assign the common heterogeneous material.
  applyMaterialField(model, mesh, material_field);

  // Apply boundary conditions.
  model.applyBC(BC::Dirichlet::FixedValue(0., _y), "Yblocked");

  model.applyBC(BC::Dirichlet::FixedValue(0., _x), "left");
  model.applyBC(BC::Dirichlet::FixedValue(0., _x), "right");

  applyInitialVelocity(model, strain_rate);
}

/// @brief Applies the velocity boundary conditions to the left and right
///        ends of the bar.
/// @param model corresponding to the cohesive zone model
/// @param applied_velocity Applied velocity at th left/right ends of the bar
/// @param current_time_step Current time step of the simulation
/// @param dt Time step size of the simulation
void applyVelocityBoundaryConditions(SolidMechanicsModelCohesive& model,
                                     Real applied_velocity,
                                     Int current_time_step,
                                     Real dt) {
  const Real time = static_cast<Real>(current_time_step) * dt;

  model.applyBC(BC::Dirichlet::FixedValue(-applied_velocity * time, _x),
                "left");

  model.applyBC(BC::Dirichlet::FixedValue(+applied_velocity * time, _x),
                "right");
}

/**
 * @brief Assign the common spatial cohesive-strength field to the
 * potential cohesive facets.
 *
 * The fracture toughness G_c is kept constant and is defined in the
 * cohesive material file.
 *
 * @param model Cohesive-zone model.
 * @param mesh Mesh of the pseudo-1D bar.
 * @param material_field Common spatial material field.
 */
void applyMaterialField(SolidMechanicsModelCohesive& model,
                        Mesh& mesh,
                        const MaterialField& material_field) {
  auto& mesh_facets = mesh.getMeshFacets();
  const auto& nodes = mesh.getNodes();

  auto& cohesive_material = model.getMaterial("cohesive");

  const Int spatial_dimension = mesh.getSpatialDimension();

  // Loop over all facet types.
  for (auto facet_type : mesh_facets.elementTypes(spatial_dimension - 1)) {
    const auto& connectivity = mesh_facets.getConnectivity(facet_type);

    if (connectivity.size() == 0) {
      continue;
    }

    // Cohesive strength stored at facet quadrature points.
    auto& sigma_c =
        cohesive_material.getArray<Real>("sigma_c", facet_type, _not_ghost);

    if (sigma_c.size() % connectivity.size() != 0) {
      throw std::runtime_error("Unexpected cohesive strength field size.");
    }

    const Int nb_quad = sigma_c.size() / connectivity.size();

    // Assign sigma_c according to the physical facet position.
    for (Int el = 0; el < connectivity.size(); ++el) {
      const Int node_0 = connectivity(el, 0);

      const Int node_1 = connectivity(el, 1);

      const Real x = 0.5 * (nodes(node_0, 0) + nodes(node_1, 0));

      const Real local_sigma_c = material_field.GetSigmac(x);

      for (Int q = 0; q < nb_quad; ++q) {
        const Int index = el * nb_quad + q;

        sigma_c(index) = local_sigma_c;
      }
    }
  }
}
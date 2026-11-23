#ifndef MAKE_BAR_HH
#define MAKE_BAR_HH

#include "bar.hh"

struct Parameters;
class MaterialField;

/**
 * @brief Builds the peridynamic bar, its points, bonds, stiffness and strength.
 *
 * The bar consists of equally spaced material points. Each point interacts
 * with every neighbouring point lying within the peridynamic horizon.
 *
 * Additional ghost points are placed beyond both ends of the bar and are used
 * to impose the boundary conditions.
 *
 * The function computes the bond geometry, partial-volume correction,
 * weighting, stiffness, local strength, breaking stretch and stable time step.
 *
 * The cohesive strength is taken from the common spatial material field so
 * that the peridynamic and cohesive-zone simulations use the same material
 * realisation.
 *
 * @param params Simulation and material parameters.
 * @param material_field Common spatial material field.
 *
 * @return Fully initialised Bar structure.
 */
Bar makeBar(const Parameters& params, const MaterialField& material_field);

#endif // MAKE_BAR_HH
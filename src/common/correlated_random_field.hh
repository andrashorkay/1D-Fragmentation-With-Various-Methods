#ifndef CORRELATED_RANDOM_FIELD_HH
#define CORRELATED_RANDOM_FIELD_HH

#include "spatial_random_field.hh"

#include <cstddef>

/**
 * @brief Generate a spatially correlated standard Gaussian random field.
 *
 * Independent Gaussian noise is generated on a fine auxiliary grid and
 * smoothed using a Gaussian kernel with the prescribed correlation length.
 */
class CorrelatedRandomField : public SpatialRandomField {
public:
  /**
   * @brief Construct a correlated random-field generator.
   *
   * @param correlation_length Spatial correlation length [m].
   * @param points_per_correlation_length Resolution of the auxiliary grid.
   */
  CorrelatedRandomField(double correlation_length,
                        std::size_t points_per_correlation_length = 8);

  /**
   * @brief Generate the correlated standard Gaussian field.
   *
   * @param positions Physical positions along the bar [m].
   * @param seed Random-number-generator seed.
   * @return Correlated standard Gaussian values at the requested positions.
   */
  std::vector<double> generate(const std::vector<double>& positions,
                               std::uint64_t seed) const override;

private:
  double correlation_length_;
  std::size_t points_per_correlation_length_;
};

#endif // CORRELATED_RANDOM_FIELD_HH
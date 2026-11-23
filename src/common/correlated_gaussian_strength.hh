#ifndef CORRELATED_GAUSSIAN_STRENGTH_HH
#define CORRELATED_GAUSSIAN_STRENGTH_HH

#include "strength_field_generator.hh"

#include <cstddef>

/**
 * @brief Generate a spatially correlated Gaussian cohesive-strength field.
 *
 * The field is obtained by generating independent Gaussian noise on a fine
 * auxiliary grid, smoothing it with a Gaussian kernel, and mapping the result
 * to the requested physical positions.
 */
class CorrelatedGaussianStrength : public StrengthFieldGenerator {
public:
  /**
   * @brief Construct a correlated Gaussian strength generator.
   *
   * @param mean_sigma_c Mean cohesive strength [Pa].
   * @param scatter Relative standard deviation of the strength field.
   * @param correlation_length Spatial correlation length [m].
   * @param points_per_correlation_length Resolution of the auxiliary grid.
   */
  CorrelatedGaussianStrength(double mean_sigma_c,
                             double scatter,
                             double correlation_length,
                             std::size_t points_per_correlation_length = 8);

  /**
   * @brief Generate cohesive strengths at the requested positions.
   *
   * @param positions Physical positions along the bar [m].
   * @param seed Random-number-generator seed.
   * @return Cohesive strength at each requested position [Pa].
   */
  std::vector<double> generate(const std::vector<double>& positions,
                               std::uint64_t seed) const override;

private:
  double mean_sigma_c_;
  double scatter_;
  double correlation_length_;
  std::size_t points_per_correlation_length_;
};

#endif // CORRELATED_GAUSSIAN_STRENGTH_HH
#ifndef GAUSSIAN_STRENGTH_DISTRIBUTION_HH
#define GAUSSIAN_STRENGTH_DISTRIBUTION_HH

#include "strength_distribution.hh"

/**
 * @brief Map a standard Gaussian random value to cohesive strength.
 *
 * The cohesive strength is computed as
 *
 * sigma_c = mean_sigma_c * (1 + scatter * z),
 *
 * where z is a standard Gaussian random value.
 */
class GaussianStrengthDistribution : public StrengthDistribution {
public:
  /**
   * @brief Construct a Gaussian cohesive-strength distribution.
   *
   * @param mean_sigma_c Mean cohesive strength [Pa].
   * @param scatter Relative standard deviation.
   */
  GaussianStrengthDistribution(double mean_sigma_c, double scatter);

  /**
   * @brief Convert a standard Gaussian value into cohesive strength.
   *
   * @param z Standard Gaussian random value.
   * @return Cohesive strength [Pa].
   */
  double transform(double z) const override;

private:
  double mean_sigma_c_;
  double scatter_;
};

#endif // GAUSSIAN_STRENGTH_DISTRIBUTION_HH
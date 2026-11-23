#ifndef WEIBULL_STRENGTH_DISTRIBUTION_HH
#define WEIBULL_STRENGTH_DISTRIBUTION_HH

#include "strength_distribution.hh"

/**
 * @brief Map a standard Gaussian random value to a Weibull-distributed
 * cohesive strength.
 *
 * The standard Gaussian value z is first transformed to a uniform
 * probability u using the standard normal cumulative distribution.
 * The Weibull inverse cumulative distribution is then applied:
 *
 * sigma_c = scale * (-ln(1 - u))^(1 / shape).
 */
class WeibullStrengthDistribution : public StrengthDistribution {
public:
  /**
   * @brief Construct a Weibull cohesive-strength distribution.
   *
   * @param scale Weibull scale parameter [Pa].
   * @param shape Weibull shape parameter.
   */
  WeibullStrengthDistribution(double scale, double shape);

  /**
   * @brief Convert a standard Gaussian value into cohesive strength.
   *
   * @param z Standard Gaussian random value.
   * @return Cohesive strength [Pa].
   */
  double transform(double z) const override;

private:
  double scale_;
  double shape_;
};

#endif // WEIBULL_STRENGTH_DISTRIBUTION_HH
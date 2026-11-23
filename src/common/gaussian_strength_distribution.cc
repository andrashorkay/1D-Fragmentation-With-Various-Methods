#include "gaussian_strength_distribution.hh"

#include <stdexcept>

GaussianStrengthDistribution::GaussianStrengthDistribution(double mean_sigma_c,
                                                           double scatter)
  : mean_sigma_c_(mean_sigma_c), scatter_(scatter) {
  if (mean_sigma_c_ <= 0.0) {
    throw std::invalid_argument(
        "GaussianStrengthDistribution: mean_sigma_c must be positive.");
  }

  if (scatter_ < 0.0) {
    throw std::invalid_argument(
        "GaussianStrengthDistribution: scatter cannot be negative.");
  }
}

double GaussianStrengthDistribution::transform(double z) const {
  const double sigma_c = mean_sigma_c_ * (1.0 + scatter_ * z);

  if (sigma_c <= 0.0) {
    throw std::runtime_error("GaussianStrengthDistribution generated "
                             "a non-positive cohesive strength.");
  }

  return sigma_c;
}
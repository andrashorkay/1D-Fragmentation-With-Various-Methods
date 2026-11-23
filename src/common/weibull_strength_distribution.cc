#include "weibull_strength_distribution.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

WeibullStrengthDistribution::WeibullStrengthDistribution(double scale,
                                                         double shape)
  : scale_(scale), shape_(shape) {
  if (scale_ <= 0.0) {
    throw std::invalid_argument(
        "WeibullStrengthDistribution: scale must be positive.");
  }

  if (shape_ <= 0.0) {
    throw std::invalid_argument(
        "WeibullStrengthDistribution: shape must be positive.");
  }
}

double WeibullStrengthDistribution::transform(double z) const {
  // Standard normal cumulative distribution:
  // u = Phi(z).
  double u = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));

  // Prevent log(0) or exactly zero probability.
  const double epsilon = std::numeric_limits<double>::epsilon();

  u = std::clamp(u, epsilon, 1.0 - epsilon);

  // Weibull inverse cumulative distribution:
  //
  // sigma_c = scale * [-ln(1-u)]^(1/shape).
  const double sigma_c = scale_ * std::pow(-std::log(1.0 - u), 1.0 / shape_);

  if (sigma_c <= 0.0) {
    throw std::runtime_error("WeibullStrengthDistribution generated "
                             "a non-positive cohesive strength.");
  }

  return sigma_c;
}
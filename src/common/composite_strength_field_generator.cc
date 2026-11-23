#include "composite_strength_field_generator.hh"

#include <stdexcept>
#include <utility>
#include <vector>

CompositeStrengthFieldGenerator::CompositeStrengthFieldGenerator(
    std::unique_ptr<SpatialRandomField> spatial_field,
    std::unique_ptr<StrengthDistribution> distribution)
  : spatial_field_(std::move(spatial_field)),
    distribution_(std::move(distribution)) {
  if (!spatial_field_) {
    throw std::invalid_argument(
        "CompositeStrengthFieldGenerator: spatial field cannot be null.");
  }

  if (!distribution_) {
    throw std::invalid_argument(
        "CompositeStrengthFieldGenerator: distribution cannot be null.");
  }
}

std::vector<double>
CompositeStrengthFieldGenerator::generate(const std::vector<double>& positions,
                                          std::uint64_t seed) const {
  const std::vector<double> latent_field =
      spatial_field_->generate(positions, seed);

  if (latent_field.size() != positions.size()) {
    throw std::runtime_error(
        "CompositeStrengthFieldGenerator: spatial field returned "
        "an unexpected number of values.");
  }

  std::vector<double> strengths(latent_field.size());

  for (std::size_t i = 0; i < latent_field.size(); ++i) {
    strengths[i] = distribution_->transform(latent_field[i]);
  }

  return strengths;
}
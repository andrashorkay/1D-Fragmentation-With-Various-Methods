#include "independent_random_field.hh"

#include <random>
#include <vector>

std::vector<double>
IndependentRandomField::generate(const std::vector<double>& positions,
                                 std::uint64_t seed) const {
  std::vector<double> field(positions.size(), 0.0);

  if (positions.empty()) {
    return field;
  }

  std::mt19937 generator(static_cast<std::mt19937::result_type>(seed));

  std::normal_distribution<double> normal_distribution(0.0, 1.0);

  for (double& value : field) {
    value = normal_distribution(generator);
  }

  return field;
}
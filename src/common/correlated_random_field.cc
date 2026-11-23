#include "correlated_random_field.hh"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

CorrelatedRandomField::CorrelatedRandomField(
    double correlation_length, std::size_t points_per_correlation_length)
  : correlation_length_(correlation_length),
    points_per_correlation_length_(points_per_correlation_length) {
  if (correlation_length_ <= 0.0) {
    throw std::invalid_argument(
        "CorrelatedRandomField: correlation_length must be positive.");
  }

  if (points_per_correlation_length_ == 0) {
    throw std::invalid_argument(
        "CorrelatedRandomField: points_per_correlation_length "
        "must be positive.");
  }
}

std::vector<double>
CorrelatedRandomField::generate(const std::vector<double>& positions,
                                std::uint64_t seed) const {
  if (positions.empty()) {
    return {};
  }

  std::mt19937 generator(static_cast<std::mt19937::result_type>(seed));

  std::normal_distribution<double> normal_distribution(0.0, 1.0);

  const double spacing =
      correlation_length_ / static_cast<double>(points_per_correlation_length_);

  const auto [min_it, max_it] =
      std::minmax_element(positions.begin(), positions.end());

  const double physical_x_min = *min_it;
  const double physical_x_max = *max_it;

  // Buffer the auxiliary grid to avoid smoothing edge effects.
  const double margin = 8.0 * correlation_length_;

  const double x_min = physical_x_min - margin;
  const double x_max = physical_x_max + margin;

  const std::size_t n_grid =
      static_cast<std::size_t>(std::ceil((x_max - x_min) / spacing)) + 1;

  // Independent standard Gaussian noise.
  std::vector<double> raw_noise(n_grid);

  for (double& value : raw_noise) {
    value = normal_distribution(generator);
  }

  // Gaussian smoothing kernel truncated at +/- 3 correlation lengths.
  const int half_width =
      static_cast<int>(std::ceil(3.0 * correlation_length_ / spacing));

  std::vector<double> kernel(static_cast<std::size_t>(2 * half_width + 1));

  double sum_squared_weights = 0.0;

  for (int k = -half_width; k <= half_width; ++k) {
    const double distance = static_cast<double>(k) * spacing;

    const double weight = std::exp(-0.5 * distance * distance /
                                   (correlation_length_ * correlation_length_));

    kernel[static_cast<std::size_t>(k + half_width)] = weight;

    sum_squared_weights += weight * weight;
  }

  // Preserve approximately unit variance after smoothing.
  const double normalisation = std::sqrt(sum_squared_weights);

  for (double& weight : kernel) {
    weight /= normalisation;
  }

  // Smooth the independent noise.
  std::vector<double> smooth_noise(n_grid, 0.0);

  for (std::size_t i = 0; i < n_grid; ++i) {
    double value = 0.0;

    for (int k = -half_width; k <= half_width; ++k) {
      const long j = static_cast<long>(i) + static_cast<long>(k);

      if (j < 0 || j >= static_cast<long>(n_grid)) {
        continue;
      }

      value += kernel[static_cast<std::size_t>(k + half_width)] *
               raw_noise[static_cast<std::size_t>(j)];
    }

    smooth_noise[i] = value;
  }

  // Interpolate the smoothed field onto the requested positions.
  std::vector<double> field(positions.size(), 0.0);

  for (std::size_t i = 0; i < positions.size(); ++i) {
    const double grid_position = (positions[i] - x_min) / spacing;

    std::size_t left = static_cast<std::size_t>(std::floor(grid_position));

    if (left >= n_grid - 1) {
      left = n_grid - 2;
    }

    const std::size_t right = left + 1;

    const double alpha = grid_position - static_cast<double>(left);

    field[i] = (1.0 - alpha) * smooth_noise[left] + alpha * smooth_noise[right];
  }

  return field;
}
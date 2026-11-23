#include "material_field.hh"
#include "strength_field_generator.hh"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

MaterialField
MaterialField::generate(double length,
                        std::size_t n_points,
                        double G_c,
                        std::uint64_t seed,
                        const StrengthFieldGenerator& strength_generator) {
  if (length <= 0.0) {
    throw std::invalid_argument("MaterialField: length must be positive.");
  }

  if (n_points < 2) {
    throw std::invalid_argument("MaterialField: n_points must be at least 2.");
  }

  if (G_c <= 0.0) {
    throw std::invalid_argument("MaterialField: G_c must be positive.");
  }

  MaterialField material;

  material.points_.resize(n_points);

  // Physical positions spanning [-L/2, L/2].
  std::vector<double> positions(n_points);

  for (std::size_t i = 0; i < n_points; ++i) {
    positions[i] = -0.5 * length + length * static_cast<double>(i) /
                                       static_cast<double>(n_points - 1);
  }

  // Generate the spatial cohesive-strength field.
  const std::vector<double> strengths =
      strength_generator.generate(positions, seed);

  if (strengths.size() != n_points) {
    throw std::runtime_error("MaterialField: strength generator returned "
                             "an unexpected number of values.");
  }

  // Store the generated strength and constant fracture toughness.
  for (std::size_t i = 0; i < n_points; ++i) {
    if (strengths[i] <= 0.0) {
      throw std::runtime_error("MaterialField: generated cohesive strength "
                               "must be positive.");
    }

    material.points_[i].x = positions[i];

    material.points_[i].sigma_c = strengths[i];

    material.points_[i].G_c = G_c;
  }

  return material;
}

/**
 * @brief Save the material field to a file.
 */
void MaterialField::save(const std::string& filename) const {
  std::ofstream file(filename);

  if (!file) {
    throw std::runtime_error(
        "MaterialField: could not open file for writing: " + filename);
  }

  file << "# x[m] sigma_c[Pa] G_c[J/m^2]\n";

  file << std::scientific << std::setprecision(17);

  for (const MaterialPoint& point : points_) {
    file << point.x << " " << point.sigma_c << " " << point.G_c << "\n";
  }
}

/**
 * @brief Load a previously generated material field.
 */
MaterialField MaterialField::load(const std::string& filename) {
  std::ifstream file(filename);

  if (!file) {
    throw std::runtime_error(
        "MaterialField: could not open file for reading: " + filename);
  }

  MaterialField material;

  std::string line;

  while (std::getline(file, line)) {
    // Ignore comments and empty lines.
    const std::size_t first = line.find_first_not_of(" \t");

    if (first == std::string::npos || line[first] == '#') {
      continue;
    }

    std::istringstream stream(line);

    MaterialPoint point;

    if (!(stream >> point.x >> point.sigma_c >> point.G_c)) {
      throw std::runtime_error("MaterialField: invalid line in " + filename +
                               ": " + line);
    }

    material.points_.push_back(point);
  }

  if (material.points_.empty()) {
    throw std::runtime_error("MaterialField: no material points found in " +
                             filename);
  }

  // Ensure increasing physical position.
  std::sort(
      material.points_.begin(),
      material.points_.end(),
      [](const MaterialPoint& a, const MaterialPoint& b) { return a.x < b.x; });

  return material;
}

/**
 * @brief Return the cohesive strength at position x.
 */
double MaterialField::GetSigmac(double x) const {
  if (points_.empty()) {
    throw std::runtime_error("MaterialField: material field is empty.");
  }

  if (x <= points_.front().x) {
    return points_.front().sigma_c;
  }

  if (x >= points_.back().x) {
    return points_.back().sigma_c;
  }

  // Find the two material points surrounding x.
  const auto right = std::lower_bound(
      points_.begin(),
      points_.end(),
      x,
      [](const MaterialPoint& point, double value) { return point.x < value; });

  const auto left = right - 1;

  const double alpha = (x - left->x) / (right->x - left->x);

  return (1.0 - alpha) * left->sigma_c + alpha * right->sigma_c;
}

/**
 * @brief Return the fracture toughness at position x.
 */
double MaterialField::GetGc(double x) const {
  if (points_.empty()) {
    throw std::runtime_error("MaterialField: material field is empty.");
  }

  if (x <= points_.front().x) {
    return points_.front().G_c;
  }

  if (x >= points_.back().x) {
    return points_.back().G_c;
  }

  const auto right = std::lower_bound(
      points_.begin(),
      points_.end(),
      x,
      [](const MaterialPoint& point, double value) { return point.x < value; });

  const auto left = right - 1;

  const double alpha = (x - left->x) / (right->x - left->x);

  return (1.0 - alpha) * left->G_c + alpha * right->G_c;
}
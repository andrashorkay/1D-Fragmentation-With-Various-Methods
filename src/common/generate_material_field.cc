#include "composite_strength_field_generator.hh"
#include "correlated_random_field.hh"
#include "gaussian_strength_distribution.hh"
#include "independent_random_field.hh"
#include "material_field.hh"
#include "spatial_random_field.hh"
#include "strength_distribution.hh"
#include "weibull_strength_distribution.hh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct Parameters {
  std::uint64_t seed = 0;

  double rho = 0.0;
  double E = 0.0;
  double nu = 0.0;
  double sigma_c = 0.0;
  double G_c = 0.0;

  std::string spatial_model;
  std::string distribution;

  double scatter = 0.0;

  double weibull_scale = 0.0;
  double weibull_shape = 0.0;

  double bar_length = 0.0;

  std::size_t points_per_correlation_length = 0;
  std::size_t number_material_points = 0;

  std::string output_file;
};

enum class Section { None, Material, RandomField, User };

/**
 * @brief Remove leading and trailing whitespace from a string.
 *
 * @param text Input string.
 * @return Trimmed string.
 */
std::string trim(const std::string& text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");

  if (first == std::string::npos) {
    return "";
  }

  const std::size_t last = text.find_last_not_of(" \t\r\n");

  return text.substr(first, last - first + 1);
}

/**
 * @brief Read the common material-field parameters from a .dat file.
 *
 * @param filename Path to the parameter file.
 * @return Parsed parameters.
 */
Parameters readParameters(const std::string& filename) {
  std::ifstream file(filename);

  if (!file) {
    throw std::runtime_error("Could not open parameter file: " + filename);
  }

  Parameters parameters;
  Section section = Section::None;

  std::string line;

  while (std::getline(file, line)) {
    // Remove comments.
    const std::size_t comment_position = line.find('#');

    if (comment_position != std::string::npos) {
      line = line.substr(0, comment_position);
    }

    line = trim(line);

    if (line.empty()) {
      continue;
    }

    // Detect parameter blocks.
    if (line == "material common_material [") {
      section = Section::Material;
      continue;
    }

    if (line == "random_field [") {
      section = Section::RandomField;
      continue;
    }

    if (line == "user parameters [") {
      section = Section::User;
      continue;
    }

    if (line == "]") {
      section = Section::None;
      continue;
    }

    // Read key = value assignments.
    const std::size_t equal_position = line.find('=');

    if (equal_position == std::string::npos) {
      throw std::runtime_error("Invalid line in parameter file: " + line);
    }

    const std::string key = trim(line.substr(0, equal_position));

    const std::string value = trim(line.substr(equal_position + 1));

    if (section == Section::None) {
      if (key == "seed") {
        parameters.seed = std::stoull(value);
      }

    } else if (section == Section::Material) {
      if (key == "rho") {
        parameters.rho = std::stod(value);

      } else if (key == "E") {
        parameters.E = std::stod(value);

      } else if (key == "nu") {
        parameters.nu = std::stod(value);

      } else if (key == "sigma_c") {
        parameters.sigma_c = std::stod(value);

      } else if (key == "G_c") {
        parameters.G_c = std::stod(value);
      }

    } else if (section == Section::RandomField) {
      if (key == "spatial_model") {
        parameters.spatial_model = value;

      } else if (key == "distribution") {
        parameters.distribution = value;

      } else if (key == "scatter") {
        parameters.scatter = std::stod(value);

      } else if (key == "weibull_scale") {
        parameters.weibull_scale = std::stod(value);

      } else if (key == "weibull_shape") {
        parameters.weibull_shape = std::stod(value);

      } else if (key == "points_per_correlation_length") {
        parameters.points_per_correlation_length = std::stoull(value);

      } else if (key == "number_material_points") {
        parameters.number_material_points = std::stoull(value);
      }

    } else if (section == Section::User) {
      if (key == "bar_length") {
        parameters.bar_length = std::stod(value);

      } else if (key == "output_file") {
        parameters.output_file = value;
      }
    }
  }

  return parameters;
}

/**
 * @brief Validate the parameters required to generate a material field.
 *
 * @param parameters Parsed material-field parameters.
 */
void validateParameters(const Parameters& parameters) {
  if (parameters.rho <= 0.0) {
    throw std::runtime_error("rho must be positive.");
  }

  if (parameters.E <= 0.0) {
    throw std::runtime_error("E must be positive.");
  }

  if (parameters.sigma_c <= 0.0) {
    throw std::runtime_error("sigma_c must be positive.");
  }

  if (parameters.G_c <= 0.0) {
    throw std::runtime_error("G_c must be positive.");
  }

  if (parameters.bar_length <= 0.0) {
    throw std::runtime_error("bar_length must be positive.");
  }

  if (parameters.output_file.empty()) {
    throw std::runtime_error("output_file must be specified.");
  }

  if (parameters.spatial_model != "correlated" &&
      parameters.spatial_model != "independent") {
    throw std::runtime_error(
        "spatial_model must be 'correlated' or 'independent'.");
  }

  if (parameters.distribution != "gaussian" &&
      parameters.distribution != "weibull") {
    throw std::runtime_error("distribution must be 'gaussian' or 'weibull'.");
  }

  if (parameters.spatial_model == "correlated" &&
      parameters.points_per_correlation_length == 0) {
    throw std::runtime_error("points_per_correlation_length must be positive "
                             "for a correlated field.");
  }

  if (parameters.spatial_model == "independent" &&
      parameters.number_material_points < 2) {
    throw std::runtime_error("number_material_points must be at least 2 "
                             "for an independent field.");
  }

  if (parameters.distribution == "gaussian" && parameters.scatter < 0.0) {
    throw std::runtime_error("scatter cannot be negative.");
  }

  if (parameters.distribution == "weibull") {
    if (parameters.weibull_scale <= 0.0) {
      throw std::runtime_error("weibull_scale must be positive.");
    }

    if (parameters.weibull_shape <= 0.0) {
      throw std::runtime_error("weibull_shape must be positive.");
    }
  }
}

/**
 * @brief Construct the requested spatial random-field model.
 *
 * @param parameters Parsed parameters.
 * @param correlation_length Spatial correlation length [m].
 * @return Spatial random-field generator.
 */
std::unique_ptr<SpatialRandomField>
createSpatialField(const Parameters& parameters, double correlation_length) {
  if (parameters.spatial_model == "correlated") {
    return std::make_unique<CorrelatedRandomField>(
        correlation_length, parameters.points_per_correlation_length);
  }

  return std::make_unique<IndependentRandomField>();
}

/**
 * @brief Construct the requested cohesive-strength distribution.
 *
 * @param parameters Parsed parameters.
 * @return Cohesive-strength distribution.
 */
std::unique_ptr<StrengthDistribution>
createStrengthDistribution(const Parameters& parameters) {
  if (parameters.distribution == "gaussian") {
    return std::make_unique<GaussianStrengthDistribution>(parameters.sigma_c,
                                                          parameters.scatter);
  }

  return std::make_unique<WeibullStrengthDistribution>(
      parameters.weibull_scale, parameters.weibull_shape);
}

} // namespace

/**
 * @brief Generate the common material field from a parameter file.
 *
 * @param argc Number of command-line arguments.
 * @param argv Optional path to the parameter file.
 * @return Zero on successful completion.
 */
int main(int argc, char** argv) {
  try {
    const std::filesystem::path parameter_file =
        (argc > 1) ? std::filesystem::path(argv[1])
                   : std::filesystem::path("parameters.dat");

    const Parameters parameters = readParameters(parameter_file.string());

    validateParameters(parameters);

    /*
     * This fracture characteristic length is used as the
     * correlation length when the correlated spatial model
     * is selected.
     */
    const double correlation_length = parameters.E * parameters.G_c /
                                      (parameters.sigma_c * parameters.sigma_c);

    std::size_t n_points = 0;
    double field_spacing = 0.0;

    if (parameters.spatial_model == "correlated") {
      field_spacing =
          correlation_length /
          static_cast<double>(parameters.points_per_correlation_length);

      n_points = static_cast<std::size_t>(
                     std::ceil(parameters.bar_length / field_spacing)) +
                 1;

    } else {
      n_points = parameters.number_material_points;

      field_spacing = parameters.bar_length / static_cast<double>(n_points - 1);
    }

    // Construct the selected spatial random-field model.
    std::unique_ptr<SpatialRandomField> spatial_field =
        createSpatialField(parameters, correlation_length);

    // Construct the selected strength distribution.
    std::unique_ptr<StrengthDistribution> distribution =
        createStrengthDistribution(parameters);

    // Combine spatial structure and probability distribution.
    CompositeStrengthFieldGenerator strength_generator(std::move(spatial_field),
                                                       std::move(distribution));

    // Generate sigma_c(x) while keeping G_c constant.
    const MaterialField material =
        MaterialField::generate(parameters.bar_length,
                                n_points,
                                parameters.G_c,
                                parameters.seed,
                                strength_generator);

    // Write the field relative to the parameter-file location.
    const std::filesystem::path output_file =
        parameter_file.parent_path() / parameters.output_file;

    material.save(output_file.string());

    std::cout << "Material field generated successfully.\n"
              << "Parameter file:      " << parameter_file << '\n'
              << "Bar length:          " << parameters.bar_length << " m\n"
              << "Reference sigma_c:   " << parameters.sigma_c << " Pa\n"
              << "G_c:                 " << parameters.G_c << " J/m^2\n"
              << "Spatial model:       " << parameters.spatial_model << '\n'
              << "Distribution:        " << parameters.distribution << '\n'
              << "Seed:                " << parameters.seed << '\n';

    if (parameters.distribution == "gaussian") {
      std::cout << "Gaussian scatter:    " << parameters.scatter << '\n';

    } else {
      std::cout << "Weibull scale:       " << parameters.weibull_scale
                << " Pa\n"
                << "Weibull shape:       " << parameters.weibull_shape << '\n';
    }

    if (parameters.spatial_model == "correlated") {
      std::cout << "Correlation length:  " << correlation_length << " m\n"
                << "Points per l_c:      "
                << parameters.points_per_correlation_length << '\n';
    }

    std::cout << "Field spacing:       " << field_spacing << " m\n"
              << "Number of points:    " << n_points << '\n'
              << "Output file:         " << output_file << '\n';

  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';

    return 1;
  }

  return 0;
}
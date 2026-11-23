#ifndef SPATIAL_RANDOM_FIELD_HH
#define SPATIAL_RANDOM_FIELD_HH

#include <cstdint>
#include <vector>

/**
 * @brief Interface for generating a spatial random field.
 *
 * Implementations define the spatial dependence of a standard Gaussian
 * latent field evaluated at prescribed physical positions.
 */
class SpatialRandomField {
public:
  virtual ~SpatialRandomField() = default;

  /**
   * @brief Generate a standard Gaussian latent field.
   *
   * @param positions Physical positions along the bar [m].
   * @param seed Random-number-generator seed.
   * @return Standard Gaussian values at the requested positions.
   */
  virtual std::vector<double> generate(const std::vector<double>& positions,
                                       std::uint64_t seed) const = 0;
};

#endif // SPATIAL_RANDOM_FIELD_HH
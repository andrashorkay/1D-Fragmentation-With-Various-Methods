#ifndef INDEPENDENT_RANDOM_FIELD_HH
#define INDEPENDENT_RANDOM_FIELD_HH

#include "spatial_random_field.hh"

/**
 * @brief Generate independent standard Gaussian random values.
 *
 * Each requested physical position receives an independent sample from
 * a standard normal distribution. No spatial correlation is introduced.
 */
class IndependentRandomField : public SpatialRandomField {
public:
  /**
   * @brief Generate independent standard Gaussian values.
   *
   * @param positions Physical positions along the bar [m].
   * @param seed Random-number-generator seed.
   * @return Independent standard Gaussian values.
   */
  std::vector<double> generate(const std::vector<double>& positions,
                               std::uint64_t seed) const override;
};

#endif // INDEPENDENT_RANDOM_FIELD_HH
#ifndef STRENGTH_FIELD_GENERATOR_HH
#define STRENGTH_FIELD_GENERATOR_HH

#include <cstdint>
#include <vector>

/**
 * @brief Interface for generating a spatial cohesive-strength field.
 *
 * Derived classes define how local cohesive strengths are generated
 * at a prescribed set of physical positions.
 */
class StrengthFieldGenerator {
public:
  virtual ~StrengthFieldGenerator() = default;

  /**
   * @brief Generate cohesive strengths at the requested positions.
   *
   * @param positions Physical positions along the bar [m].
   * @param seed Random-number-generator seed.
   * @return Cohesive strength at each requested position [Pa].
   */
  virtual std::vector<double> generate(const std::vector<double>& positions,
                                       std::uint64_t seed) const = 0;
};

#endif // STRENGTH_FIELD_GENERATOR_HH
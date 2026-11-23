#ifndef COMPOSITE_STRENGTH_FIELD_GENERATOR_HH
#define COMPOSITE_STRENGTH_FIELD_GENERATOR_HH

#include "spatial_random_field.hh"
#include "strength_distribution.hh"
#include "strength_field_generator.hh"

#include <memory>

/**
 * @brief Generate a cohesive-strength field by combining a spatial
 * random field with a strength distribution.
 */
class CompositeStrengthFieldGenerator : public StrengthFieldGenerator {
public:
  /**
   * @brief Construct a cohesive-strength field generator.
   *
   * @param spatial_field Defines the spatial dependence of the randomness.
   * @param distribution Maps latent Gaussian values to cohesive strengths.
   */
  CompositeStrengthFieldGenerator(
      std::unique_ptr<SpatialRandomField> spatial_field,
      std::unique_ptr<StrengthDistribution> distribution);

  /**
   * @brief Generate cohesive strengths at the requested positions.
   *
   * @param positions Physical positions along the bar [m].
   * @param seed Random-number-generator seed.
   * @return Cohesive strength at each requested position [Pa].
   */
  std::vector<double> generate(const std::vector<double>& positions,
                               std::uint64_t seed) const override;

private:
  std::unique_ptr<SpatialRandomField> spatial_field_;
  std::unique_ptr<StrengthDistribution> distribution_;
};

#endif // COMPOSITE_STRENGTH_FIELD_GENERATOR_HH
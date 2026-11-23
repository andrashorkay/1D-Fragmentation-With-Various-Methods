#ifndef STRENGTH_DISTRIBUTION_HH
#define STRENGTH_DISTRIBUTION_HH

/**
 * @brief Interface for mapping a standard Gaussian random value
 * to a cohesive strength.
 */
class StrengthDistribution {
public:
  virtual ~StrengthDistribution() = default;

  /**
   * @brief Convert a standard Gaussian value into cohesive strength.
   *
   * @param z Standard Gaussian random value.
   * @return Cohesive strength [Pa].
   */
  virtual double transform(double z) const = 0;
};

#endif // STRENGTH_DISTRIBUTION_HH
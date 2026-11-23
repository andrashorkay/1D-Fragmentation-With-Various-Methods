#ifndef COUNT_FRAGMENTS_HH
#define COUNT_FRAGMENTS_HH

#include <cstddef>
#include <vector>

struct Bar;

struct FragmentData {
  // Fragment statistics after applying the filtering length.
  std::size_t num_fragments;
  std::vector<double> cut_x;
  std::vector<double> sizes;

  // Raw topology: every gap crossed by no active breakable bond is a cut.
  std::size_t raw_num_fragments;
  std::vector<double> raw_cut_x;
  std::vector<double> raw_sizes;

  // Connectivity information for every gap between physical points.
  std::vector<double> gap_x;
  std::vector<long> active_crossing_bonds;

  // Minimum fragment size used for the filtered result.
  double filter_length;
};

/**
 * @brief Finds where the centred bar has split and computes fragment sizes.
 *
 * The physical bar occupies [-L/2, L/2]. A gap between neighbouring
 * physical points is a raw cut when no active breakable bond crosses that gap.
 *
 * Cut positions are returned in the same centred coordinate system.
 *
 * Both the raw cuts and a filtered set of cuts are returned. By default,
 * filtered cuts closer together than one peridynamic horizon are merged.
 *
 * @param bar Peridynamic bar containing geometry and bond information.
 * @param health Current health of every bond, between 0 and 1.
 * @param filter_length Minimum accepted fragment length [m]. If negative,
 * the peridynamic horizon is used.
 *
 * @return FragmentData containing raw and filtered fragment information and
 * gap connectivity.
 */
FragmentData countFragments(const Bar& bar,
                            const std::vector<double>& health,
                            double filter_length = -1.0);

#endif
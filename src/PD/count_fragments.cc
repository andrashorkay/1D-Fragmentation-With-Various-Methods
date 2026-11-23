#include "count_fragments.hh"
#include "bar.hh"
#include <utility>

#include <stdexcept>
#include <vector>

#include "count_fragments.hh"

#include "bar.hh"

#include <stdexcept>
#include <utility>
#include <vector>

FragmentData countFragments(const Bar& bar,
                            const std::vector<double>& health,
                            double filter_length) {
  const std::size_t N = bar.num_points;

  if (health.size() != bar.bond.count) {
    throw std::runtime_error(
        "Bond-health vector does not match the number of bonds.");
  }

  // --------------------------------------------------------------------------
  // Physical domain.
  //
  // The bar is centred on x = 0.

  const double x_left = -0.5 * bar.L;

  const double x_right = 0.5 * bar.L;

  // --------------------------------------------------------------------------
  // Count active breakable bonds crossing each gap between physical points.

  std::vector<long> change(N, 0);

  for (std::size_t b = 0; b < health.size(); ++b) {
    if (health[b] > 0.0 && bar.bond.can_break[b]) {
      const std::size_t i = bar.bond.i[b] - bar.num_boundary;

      const std::size_t j = bar.bond.j[b] - bar.num_boundary;

      ++change[i];
      --change[j];
    }
  }

  std::vector<long> crossing(N, 0);

  if (N > 0) {
    crossing[0] = change[0];
  }

  for (std::size_t i = 1; i < N; ++i) {
    crossing[i] = crossing[i - 1] + change[i];
  }

  // --------------------------------------------------------------------------
  // Store physical gap positions and active connectivity.

  std::vector<double> gap_x;
  std::vector<long> active_crossing_bonds;

  if (N > 1) {
    gap_x.reserve(N - 1);
    active_crossing_bonds.reserve(N - 1);
  }

  for (std::size_t gap = 0; gap + 1 < N; ++gap) {
    const std::size_t left = bar.num_boundary + gap;

    const std::size_t right = left + 1;

    gap_x.push_back(0.5 * (bar.x[left] + bar.x[right]));

    active_crossing_bonds.push_back(crossing[gap]);
  }

  // --------------------------------------------------------------------------
  // Raw cuts.
  //
  // A cut exists where no active breakable bond crosses the gap.

  std::vector<double> raw_cut_x;

  for (std::size_t gap = 0; gap < active_crossing_bonds.size(); ++gap) {
    if (active_crossing_bonds[gap] == 0) {
      raw_cut_x.push_back(gap_x[gap]);
    }
  }

  // --------------------------------------------------------------------------
  // Raw fragment sizes.

  std::vector<double> raw_sizes;

  raw_sizes.reserve(raw_cut_x.size() + 1);

  double previous = x_left;

  for (const double cut : raw_cut_x) {
    raw_sizes.push_back(cut - previous);

    previous = cut;
  }

  raw_sizes.push_back(x_right - previous);

  // --------------------------------------------------------------------------
  // Filter cuts using either the requested filtering length or the horizon.

  const double min_size = filter_length < 0.0 ? bar.horizon : filter_length;

  if (min_size < 0.0) {
    throw std::runtime_error("Fragment filtering length must be non-negative.");
  }

  std::vector<double> cut_x;

  previous = x_left;

  for (const double cut : raw_cut_x) {
    if (cut - previous >= min_size) {
      cut_x.push_back(cut);

      previous = cut;
    }
  }

  // Remove the final cut if it would leave a fragment smaller than the
  // filtering length at the right end of the bar.

  while (!cut_x.empty() && x_right - cut_x.back() < min_size) {
    cut_x.pop_back();
  }

  // --------------------------------------------------------------------------
  // Filtered fragment sizes.

  std::vector<double> sizes;

  sizes.reserve(cut_x.size() + 1);

  previous = x_left;

  for (const double cut : cut_x) {
    sizes.push_back(cut - previous);

    previous = cut;
  }

  sizes.push_back(x_right - previous);

  // --------------------------------------------------------------------------
  // Return raw, filtered and connectivity information.

  FragmentData result;

  result.num_fragments = cut_x.size() + 1;

  result.cut_x = std::move(cut_x);

  result.sizes = std::move(sizes);

  result.raw_num_fragments = raw_cut_x.size() + 1;

  result.raw_cut_x = std::move(raw_cut_x);

  result.raw_sizes = std::move(raw_sizes);

  result.gap_x = std::move(gap_x);

  result.active_crossing_bonds = std::move(active_crossing_bonds);

  result.filter_length = min_size;

  return result;
}
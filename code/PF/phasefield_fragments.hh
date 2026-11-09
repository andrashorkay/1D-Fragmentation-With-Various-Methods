#pragma once

#include <aka_common.hh>

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "coupler_solid_phasefield.hh"
#include "group_manager.hh"
#include "mesh_iterators.hh"
#include "phase_field_element_filter.hh"

using namespace akantu;

class PhaseFieldFragmentManager : public GroupManager {
public:
  PhaseFieldFragmentManager(PhaseFieldModel& model,
                            const Real& damage_limit,
                            const ID& id)
    : GroupManager(model.getMesh(), id), model(model),
      damage_limit(damage_limit) {}

  // ------------------------------------------------------------------------ //
  // Original Akantu-style 2D connected-component count.
  //
  // Keep this because getFragment() relies on the element groups created here.
  // Do not use this count as the physical fragment count for the pseudo-1D bar.

  Int countFragments(const std::string& prefix = "fragment") {
    const Int dim = model.getSpatialDimension();

    nb_fragments = createClusters(
        dim, prefix, PhaseFieldElementFilter(model, damage_limit));

    cluster_prefix = prefix;

    return nb_fragments;
  }

  // ------------------------------------------------------------------------ //
  // Physical fragment count for the pseudo-1D bar.
  //
  // A crack is a contiguous interval along x for which the damage exceeds the
  // chosen damage threshold.
  //
  // Number of fragments = number of crack intervals + 1.
  //
  // At each x coordinate, the minimum damage through the bar height is used.
  // Therefore, a position is considered broken only if the full cross-section
  // has reached the damage threshold.

  Int countFragments1D() {
    const auto profile = build1DProfile();

    if (profile.empty()) {
      nb_fragments_1d = 0;
      return nb_fragments_1d;
    }

    Int nb_cracks = 0;
    bool inside_crack = false;

    for (const auto& point : profile) {
      const Real damage = point.second;
      const bool broken = damage >= damage_limit;

      if (broken && !inside_crack) {
        ++nb_cracks;
      }

      inside_crack = broken;
    }

    nb_fragments_1d = nb_cracks + 1;

    return nb_fragments_1d;
  }

  // ------------------------------------------------------------------------ //
  // Compute the 1D fragment sizes.
  //
  // Each diffuse crack is represented by the centre of its damaged interval.
  // Fragment sizes are then the distances between consecutive crack centres
  // and the two ends of the bar.

  std::vector<Real> computeFragmentSizes1D() const {
    const auto profile = build1DProfile();

    std::vector<Real> fragment_sizes;

    if (profile.empty()) {
      return fragment_sizes;
    }

    std::vector<Real> crack_centres;

    bool inside_crack = false;
    Real crack_start = 0.0;
    Real crack_end = 0.0;

    for (const auto& point : profile) {
      const Real x = point.first;
      const Real damage = point.second;
      const bool broken = damage >= damage_limit;

      if (broken && !inside_crack) {
        crack_start = x;
        crack_end = x;
        inside_crack = true;
      } else if (broken) {
        crack_end = x;
      } else if (inside_crack) {
        crack_centres.push_back(0.5 * (crack_start + crack_end));
        inside_crack = false;
      }
    }

    if (inside_crack) {
      crack_centres.push_back(0.5 * (crack_start + crack_end));
    }

    Real fragment_start = profile.front().first;

    for (const Real crack_centre : crack_centres) {
      fragment_sizes.push_back(crack_centre - fragment_start);
      fragment_start = crack_centre;
    }

    fragment_sizes.push_back(profile.back().first - fragment_start);

    return fragment_sizes;
  }

  // ------------------------------------------------------------------------ //

  ElementGroup& getFragment(Int fragment_id) {
    return getElementGroup(cluster_prefix + "_" + std::to_string(fragment_id));
  }

  // ------------------------------------------------------------------------ //

  Int getNbFragments() const {
    return nb_fragments;
  }

  // ------------------------------------------------------------------------ //

  Int getNbFragments1D() const {
    return nb_fragments_1d;
  }

  // ------------------------------------------------------------------------ //

  Real computeLowDamageEnergy(const Real& damage_limit) {
    PhaseFieldElementFilter element_filter(model, damage_limit);

    const std::string name = "low_damage_" + std::to_string(damage_limit);

    auto& group = createElementGroup(name);

    for_each_element(mesh, [&](auto&& element) {
      if (element_filter(element)) {
        group.add(element);
      }
    });

    return model.getEnergy(name);
  }

private:
  // ------------------------------------------------------------------------ //
  // Collapse the pseudo-2D mesh onto a 1D damage profile.
  //
  // There are several nodes with the same x coordinate. For every x, keep the
  // minimum damage. This means a crack is counted only when the damaged zone
  // cuts through the complete height of the bar.

  std::vector<std::pair<Real, Real>> build1DProfile() const {
    const auto& mesh = model.getMesh();
    const auto& coordinates = mesh.getNodes();
    const auto& damage = model.getDamage();

    std::vector<std::pair<Real, Real>> nodal_data;

    nodal_data.reserve(mesh.getNbNodes());

    for (Int node = 0; node < mesh.getNbNodes(); ++node) {
      nodal_data.emplace_back(coordinates(node, 0), damage(node));
    }

    std::sort(nodal_data.begin(),
              nodal_data.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<Real, Real>> profile;

    std::size_t i = 0;

    while (i < nodal_data.size()) {
      const Real x = nodal_data[i].first;
      Real min_damage = nodal_data[i].second;

      std::size_t j = i + 1;

      while (j < nodal_data.size() &&
             Math::are_float_equal(nodal_data[j].first, x)) {
        min_damage = std::min(min_damage, nodal_data[j].second);
        ++j;
      }

      profile.emplace_back(x, min_damage);

      i = j;
    }

    return profile;
  }

private:
  PhaseFieldModel& model;
  Real damage_limit;

  Int nb_fragments{0};
  Int nb_fragments_1d{0};

  std::string cluster_prefix{"fragment"};
};
cmake_minimum_required(VERSION 3.12)

project(fragmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Akantu REQUIRED)


# ============================================================================ #
# Material Field Generator

add_executable(
  generate_material_field

  src/common/generate_material_field.cc
  src/common/material_field.cc

  src/common/correlated_random_field.cc
  src/common/independent_random_field.cc

  src/common/gaussian_strength_distribution.cc
  src/common/weibull_strength_distribution.cc

  src/common/composite_strength_field_generator.cc
)

target_include_directories(
  generate_material_field
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common
)


# ============================================================================ #
# Cohesive Zone Model

add_akantu_simulation(
  fragmentation_czm

  src/CZM/main.cc
  src/CZM/generate_model.cc
  src/CZM/model_functions.cc

  src/common/generate_mesh.cc
  src/common/diagnostic_writer.cc
  src/common/material_field.cc
)

target_include_directories(
  fragmentation_czm
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CZM
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common
)


# ============================================================================ #
# Phase Field

add_akantu_simulation(
  fragmentation_pf

  src/PF/main.cc
  src/PF/generate_model.cc
  src/PF/model_functions.cc

  src/common/generate_mesh.cc
  src/common/diagnostic_writer.cc
)

target_include_directories(
  fragmentation_pf
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/PF
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common
)


# ============================================================================ #
# Peridynamics

add_executable(
  fragmentation_pd

  src/PD/main.cc
  src/PD/parameters.cc
  src/PD/make_bar.cc
  src/PD/simulate_bar.cc
  src/PD/bond_health.cc
  src/PD/count_fragments.cc
  src/PD/fragment_size_theory.cc
  src/PD/pd_diagnostic_writer.cc

  src/common/material_field.cc
)

target_include_directories(
  fragmentation_pd
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/PD
  ${CMAKE_CURRENT_SOURCE_DIR}/src/common
)
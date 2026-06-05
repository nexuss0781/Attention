cmake_minimum_required(VERSION 3.25)
project(SMAO VERSION 1.0.0 LANGUAGES CXX)

# C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic)
    # Avoid -ffast-math by default; use it only in specific optimized builds
endif()

# Find required packages
find_package(Eigen3 3.4 REQUIRED CONFIG)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Catch2 3.4 QUIET)

# Phase 1 subdirectory
add_subdirectory(Phase-1)

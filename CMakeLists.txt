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

# Enable testing at root level so CTest can find tests
set(BUILD_TESTING ON CACHE BOOL "Enable testing" FORCE)
enable_testing()

# Find GoogleTest locally, or fetch it from GitHub if not found
find_package(GTest 1.10 CONFIG QUIET)
if(NOT GTEST_FOUND)
    find_package(GTest QUIET)
endif()

if(NOT GTEST_FOUND)
    message(STATUS "GoogleTest not found locally, fetching from GitHub...")
    # Disable building GoogleTest's own internal tests and samples
    set(gtest_build_tests OFF CACHE BOOL "Build GTest tests" FORCE)
    set(gtest_build_samples OFF CACHE BOOL "Build GTest samples" FORCE)
    set(gmock_build_tests OFF CACHE BOOL "Build GMock tests" FORCE)

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # Prevent overriding compiler/linker settings on Windows/MSVC
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(GTEST_FOUND TRUE)
endif()

# Phase 1 subdirectory
add_subdirectory(Phase-1)

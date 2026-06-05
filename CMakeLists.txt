cmake_minimum_required(VERSION 3.25)
project(SMAO_Phase1 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for performance and standards compliance
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -ffast-math -Wall -Wextra -Wpedantic)
    add_compile_options(-fPIC)
elseif(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4 /permissive-)
endif()

# Find dependencies
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Source files
set(SMAO_SOURCES
    src/core/types.cpp
    src/core/exact_decomposition.cpp
    src/core/metric_assembly.cpp
    src/core/whiten_coordinates.cpp
    src/core/anisotropic_distance.cpp
    src/core/numerical_guards.cpp
    src/core/phase1_forward.cpp
)

# Header files (for IDE support)
set(SMAO_HEADERS
    include/smao_phase1/smao_phase1.h
    include/smao_phase1/core/types.h
    include/smao_phase1/core/exact_decomposition.h
    include/smao_phase1/core/metric_assembly.h
    include/smao_phase1/core/whiten_coordinates.h
    include/smao_phase1/core/anisotropic_distance.h
    include/smao_phase1/core/numerical_guards.h
    include/smao_phase1/core/phase1_forward.h
)

# Create static library
add_library(smao_phase1_static STATIC ${SMAO_SOURCES} ${SMAO_HEADERS})
target_include_directories(smao_phase1_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(smao_phase1_static
    PUBLIC
        Eigen3::Eigen
        BLAS::BLAS
        LAPACK::LAPACK
)
set_target_properties(smao_phase1_static PROPERTIES OUTPUT_NAME smao_phase1)

# Create shared library
add_library(smao_phase1_shared SHARED ${SMAO_SOURCES} ${SMAO_HEADERS})
target_include_directories(smao_phase1_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(smao_phase1_shared
    PUBLIC
        Eigen3::Eigen
        BLAS::BLAS
        LAPACK::LAPACK
)
set_target_properties(smao_phase1_shared PROPERTIES OUTPUT_NAME smao_phase1)

# Tests
enable_testing()
add_executable(smao_phase1_tests
    tests/test_exact_decomposition.cpp
    tests/test_metric_assembly.cpp
    tests/test_whiten_coordinates.cpp
    tests/test_anisotropic_distance.cpp
    tests/test_numerical_guards.cpp
    tests/test_phase1_forward.cpp
    tests/test_algebraic_equivalence.cpp
    tests/test_gradient_accuracy.cpp
)
target_link_libraries(smao_phase1_tests
    smao_phase1_static
    GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(smao_phase1_tests)

# Benchmarks
add_executable(smao_phase1_benchmarks
    benchmarks/benchmark_throughput.cpp
)
target_link_libraries(smao_phase1_benchmarks
    smao_phase1_static
)

# Installation
include(GNUInstallDirs)
install(TARGETS smao_phase1_static smao_phase1_shared
    EXPORT smao_phase1-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT smao_phase1-targets
    FILE smao_phase1-targets.cmake
    NAMESPACE smao::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/smao_phase1
)

# Phase 1: Kernel Geometry & Metric Space
# Spectral Multipole Attention Operator (SMAO)

# Source files
set(PHASE1_SOURCES
    src/exact_decomposition.cpp
    src/metric_assembly.cpp
    src/whiten_coordinates.cpp
    src/anisotropic_distance.cpp
    src/phase1_forward.cpp
)

set(PHASE1_HEADERS
    include/smao_phase1.h
    include/phase1_types.h
    include/exact_decomposition.h
    include/metric_assembly.h
    include/whiten_coordinates.h
    include/anisotropic_distance.h
    include/phase1_forward.h
    include/numerical_stability.h
)

# Static library
add_library(smao_phase1_static STATIC ${PHASE1_SOURCES})
target_include_directories(smao_phase1_static
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(smao_phase1_static
    PUBLIC
        Eigen3::Eigen
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
)
set_target_properties(smao_phase1_static PROPERTIES
    OUTPUT_NAME smao_phase1
)

# Shared library
add_library(smao_phase1_shared SHARED ${PHASE1_SOURCES})
target_include_directories(smao_phase1_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(smao_phase1_shared
    PUBLIC
        Eigen3::Eigen
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
)
set_target_properties(smao_phase1_shared PROPERTIES
    OUTPUT_NAME smao_phase1
)

# Tests (if Catch2 is available)
if(Catch2_FOUND)
    set(TEST_SOURCES
        tests/test_exact_decomposition.cpp
        tests/test_whitening_isometry.cpp
        tests/test_anisotropic_consistency.cpp
        tests/test_overflow_guard.cpp
        tests/test_near_singular_metric.cpp
        tests/test_adversarial_orthogonal.cpp
        tests/test_gradient_verification.cpp
        tests/test_distribution_preservation.cpp
    )

    add_executable(phase1_tests ${TEST_SOURCES})
    target_include_directories(phase1_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(phase1_tests PRIVATE smao_phase1_static Catch2::Catch2WithMain)
    
    enable_testing()
    add_test(NAME Phase1Tests COMMAND phase1_tests)
endif()

# Benchmarks
add_executable(phase1_benchmarks benchmarks/phase1_benchmarks.cpp)
target_include_directories(phase1_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(phase1_benchmarks PRIVATE smao_phase1_static)

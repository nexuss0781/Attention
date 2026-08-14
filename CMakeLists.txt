cmake_minimum_required(VERSION 3.20)

project(Attention VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ATTENTION_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ATTENTION_ENABLE_NATIVE "Enable host-native SIMD instruction tuning" ON)
option(ATTENTION_ENABLE_OPENMP "Enable OpenMP parallel dense kernels" ON)
option(ATTENTION_ENABLE_CBLAS "Enable optimized-provider CBLAS row-major GEMM" OFF)
option(ATTENTION_BUILD_TESTS "Build Attention tests" ON)
option(ATTENTION_BUILD_BENCHMARKS "Build Attention benchmarks" ON)
option(ATTENTION_ENABLE_PERFORMANCE_TESTS "Register strict performance regression tests" OFF)

include(FetchContent)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Eigen3::Eigen target is required")
endif()

if(ATTENTION_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP enabled: ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "OpenMP not found; using single-threaded Eigen kernels")
    endif()
endif()

include(CheckIncludeFileCXX)
check_include_file_cxx("cblas.h" ATTENTION_HAS_CBLAS_HEADER)
set(ATTENTION_USE_CBLAS OFF)
if(ATTENTION_ENABLE_CBLAS AND ATTENTION_HAS_CBLAS_HEADER)
    find_package(BLAS QUIET)
    string(TOLOWER "${BLAS_LIBRARIES}" ATTENTION_BLAS_LIBRARIES_LOWER)
    if(BLAS_FOUND AND ATTENTION_BLAS_LIBRARIES_LOWER MATCHES "openblas|blis|mkl")
        set(ATTENTION_USE_CBLAS ON)
        message(STATUS "Optimized CBLAS enabled: ${BLAS_LIBRARIES}")
    else()
        message(STATUS "No recognized optimized CBLAS provider; using Eigen GEMM")
    endif()
endif()

if(ATTENTION_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT TARGET GTest::gtest_main)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra -Wpedantic)
    if(ATTENTION_ENABLE_NATIVE)
        add_compile_options(-march=native -mtune=native)
    endif()
elseif(MSVC)
    add_compile_options(/O2 /W4 /permissive-)
endif()

if(ATTENTION_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

set(ATTENTION_SOURCES
    src/transformer_config.cpp
    src/tensor.cpp
    src/parameter_store.cpp
    src/feed_forward.cpp
    src/normalization.cpp
    src/final_output.cpp
    src/vocabulary_projection.cpp
    src/transformer_block.cpp
    src/residual_connection.cpp
    src/token_embedding.cpp
    src/positional_encoding.cpp
    src/qkv_projection.cpp
    src/causal_mask.cpp
    src/linear_attention.cpp
    src/linear_attention_stream.cpp
    src/smao_linear_boundary.cpp
    src/core/types.cpp
    src/core/exact_decomposition.cpp
    src/core/metric_assembly.cpp
    src/core/whiten_coordinates.cpp
    src/core/anisotropic_distance.cpp
    src/core/numerical_guards.cpp
    src/core/phase1_forward.cpp
    src/c_api.cpp
)

set(ATTENTION_HEADERS
    include/attention/transformer_config.h
    include/attention/tensor.h
    include/attention/parameter_store.h
    include/attention/feed_forward.h
    include/attention/normalization.h
    include/attention/final_output.h
    include/attention/vocabulary_projection.h
    include/attention/transformer_block.h
    include/attention/residual_connection.h
    include/attention/token_embedding.h
    include/attention/positional_encoding.h
    include/attention/qkv_projection.h
    include/attention/causal_mask.h
    include/attention/linear_attention.h
    include/attention/smao_linear_boundary.h
    include/smao_phase1/smao_phase1.h
    include/smao_phase1/core/types.h
    include/smao_phase1/core/exact_decomposition.h
    include/smao_phase1/core/metric_assembly.h
    include/smao_phase1/core/whiten_coordinates.h
    include/smao_phase1/core/anisotropic_distance.h
    include/smao_phase1/core/numerical_guards.h
    include/smao_phase1/core/phase1_forward.h
)

function(configure_attention_target target)
    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${target} PUBLIC Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(ATTENTION_USE_CBLAS)
        target_compile_definitions(${target} PRIVATE SMAO_HAS_CBLAS=1)
        target_link_libraries(${target} PUBLIC ${BLAS_LIBRARIES})
    endif()
    target_compile_features(${target} PUBLIC cxx_std_20)
endfunction()

add_library(attention_static STATIC ${ATTENTION_SOURCES} ${ATTENTION_HEADERS})
configure_attention_target(attention_static)
set_target_properties(attention_static PROPERTIES OUTPUT_NAME attention)

add_library(attention_shared SHARED ${ATTENTION_SOURCES} ${ATTENTION_HEADERS})
configure_attention_target(attention_shared)
set_target_properties(attention_shared PROPERTIES OUTPUT_NAME attention VERSION ${PROJECT_VERSION} SOVERSION 1)

if(ATTENTION_BUILD_TESTS)
    enable_testing()
    add_executable(attention_tests
        tests/test_exact_decomposition.cpp
        tests/test_metric_assembly.cpp
        tests/test_whiten_coordinates.cpp
        tests/test_anisotropic_distance.cpp
        tests/test_numerical_guards.cpp
        tests/test_phase1_forward.cpp
        tests/test_algebraic_equivalence.cpp
        tests/test_gradient_accuracy.cpp
        tests/test_c_api.cpp
        tests/test_transformer_config.cpp
        tests/test_tensor_parameter.cpp
        tests/test_feed_forward.cpp
        tests/test_normalization.cpp
        tests/test_final_output.cpp
        tests/test_vocabulary_projection.cpp
        tests/test_transformer_block.cpp
        tests/test_residual_connection.cpp
        tests/test_embedding_positional.cpp
        tests/test_qkv_causal.cpp
        tests/test_linear_attention.cpp
        tests/test_linear_attention_stream.cpp
        tests/test_smao_linear_boundary.cpp
    )
    target_link_libraries(attention_tests PRIVATE attention_static GTest::gtest_main)
    target_compile_features(attention_tests PRIVATE cxx_std_20)
    include(GoogleTest)
    gtest_discover_tests(attention_tests)
endif()

if(ATTENTION_BUILD_BENCHMARKS)
    add_executable(attention_benchmarks benchmarks/benchmark_throughput.cpp)
    target_link_libraries(attention_benchmarks PRIVATE attention_static)
    target_compile_features(attention_benchmarks PRIVATE cxx_std_20)

    add_executable(attention_linear_benchmarks benchmarks/benchmark_linear_attention.cpp)
    target_link_libraries(attention_linear_benchmarks PRIVATE attention_static)
    target_compile_features(attention_linear_benchmarks PRIVATE cxx_std_20)

    add_executable(attention_stream_benchmarks benchmarks/benchmark_linear_stream.cpp)
    target_link_libraries(attention_stream_benchmarks PRIVATE attention_static)
    target_compile_features(attention_stream_benchmarks PRIVATE cxx_std_20)
    if(ATTENTION_ENABLE_PERFORMANCE_TESTS)
        add_test(NAME AttentionPerformanceRegression
                 COMMAND attention_benchmarks --full --strict --repeats 5)
        set_tests_properties(AttentionPerformanceRegression PROPERTIES TIMEOUT 300)
    endif()
elseif(ATTENTION_ENABLE_PERFORMANCE_TESTS)
    message(FATAL_ERROR "ATTENTION_ENABLE_PERFORMANCE_TESTS requires ATTENTION_BUILD_BENCHMARKS=ON")
endif()

include(GNUInstallDirs)
install(TARGETS attention_static attention_shared
    EXPORT attention-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT attention-targets
    FILE attention-targets.cmake
    NAMESPACE attention::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/attention
)

# InstallDependencies.cmake
# Helper module to install and find dependencies

include(FetchContent)

# Set FetchContent to not update already populated content
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Function to install Eigen3 if not found
function(install_eigen3)
    message(STATUS "Eigen3 not found. Fetching from source...")
    
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
        set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR} CACHE PATH "Eigen3 include directory" FORCE)
        set(EIGEN3_FOUND TRUE CACHE BOOL "Eigen3 found" FORCE)
    endif()
    
    message(STATUS "Eigen3 fetched successfully at ${EIGEN3_INCLUDE_DIR}")
endfunction()

# Function to install Google Test if not found
function(install_gtest)
    message(STATUS "Google Test not found. Fetching from source...")
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    
    # Prevent GoogleTest from overriding compiler options
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
    
    message(STATUS "Google Test fetched successfully")
endfunction()

# Function to find or install BLAS/LAPACK
function(find_or_install_blas_lapack)
    # First try to find system BLAS/LAPACK
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "System BLAS and LAPACK found")
        message(STATUS "  BLAS libraries: ${BLAS_LIBRARIES}")
        message(STATUS "  LAPACK libraries: ${LAPACK_LIBRARIES}")
        return()
    endif()
    
    # Try to find OpenBLAS specifically
    find_library(OPENBLAS_LIB openblas PATHS /usr/lib /usr/local/lib /opt/OpenBLAS/lib)
    find_path(OPENBLAS_INCLUDE cblas.h PATHS /usr/include /usr/local/include /opt/OpenBLAS/include)
    
    if(OPENBLAS_LIB AND OPENBLAS_INCLUDE)
        message(STATUS "OpenBLAS found")
        set(BLAS_FOUND TRUE CACHE BOOL "BLAS found" FORCE)
        set(BLAS_LIBRARIES ${OPENBLAS_LIB} CACHE FILEPATH "BLAS libraries" FORCE)
        set(BLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE} CACHE PATH "BLAS include directories" FORCE)
        set(LAPACK_FOUND TRUE CACHE BOOL "LAPACK found" FORCE)
        set(LAPACK_LIBRARIES ${OPENBLAS_LIB} CACHE FILEPATH "LAPACK libraries" FORCE)
        return()
    endif()
    
    message(WARNING "Neither system BLAS/LAPACK nor OpenBLAS found!")
    message(WARNING "Please install BLAS/LAPACK manually:")
    message(WARNING "  Ubuntu/Debian: sudo apt-get install libopenblas-dev liblapack-dev")
    message(WARNING "  macOS: brew install openblas lapack")
    message(WARNING "  CentOS/RHEL: sudo yum install openblas-devel lapack-devel")
endfunction()

# Main function to install all dependencies
function(install_all_dependencies)
    message(STATUS "========================================")
    message(STATUS "Installing SMAO Phase 1 Dependencies")
    message(STATUS "========================================")
    
    # Find or install BLAS/LAPACK first (system-level)
    find_or_install_blas_lapack()
    
    # Find or install Eigen3
    find_package(Eigen3 3.4 QUIET)
    if(NOT Eigen3_FOUND)
        install_eigen3()
    else()
        message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
    endif()
    
    # Find or install Google Test
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        install_gtest()
    else()
        message(STATUS "Google Test found")
    endif()
    
    message(STATUS "========================================")
    message(STATUS "Dependency Installation Complete")
    message(STATUS "========================================")
endfunction()

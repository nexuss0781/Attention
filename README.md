# SMAO Phase 1: Kernel Geometry & Metric Space

Complete C++20 implementation of Phase 1 of the Spectral Multipole Attention Operator (SMAO).

## Overview

Phase 1 establishes the algebraic and geometric foundation for the attention mechanism by:
- Eliminating the softmax nonlinearity through exact Gaussian kernel decomposition
- Introducing a learned anisotropic metric that re-scales the embedding space
- Producing validated, numerically stable coordinates for subsequent phases

## Project Structure

```
.
├── CMakeLists.txt              # Main build configuration
├── README.md                   # This file
├── include/
│   └── smao_phase1/
│       ├── smao_phase1.h       # Public C API
│       └── core/
│           ├── types.h         # Core type definitions
│           ├── exact_decomposition.h
│           ├── metric_assembly.h
│           ├── whiten_coordinates.h
│           ├── anisotropic_distance.h
│           ├── numerical_guards.h
│           └── phase1_forward.h
├── src/
│   ├── core/
│   │   ├── types.cpp
│   │   ├── exact_decomposition.cpp    # Algorithm 1
│   │   ├── metric_assembly.cpp        # Algorithm 2
│   │   ├── whiten_coordinates.cpp     # Algorithm 3
│   │   ├── anisotropic_distance.cpp   # Algorithm 4
│   │   ├── numerical_guards.cpp       # Contracts 3.1-3.3
│   │   └── phase1_forward.cpp         # Algorithm 5
│   └── c_api.cpp                      # C API implementation
├── tests/
│   ├── test_exact_decomposition.cpp   # Test 1.1, 1.4, 1.6
│   ├── test_metric_assembly.cpp       # Test 1.2, 1.5
│   ├── test_whiten_coordinates.cpp
│   ├── test_anisotropic_distance.cpp  # Test 1.3, 1.8
│   ├── test_numerical_guards.cpp      # Contracts 3.1-3.3
│   ├── test_phase1_forward.cpp        # Integration tests
│   └── test_gradient_accuracy.cpp     # Test 1.7
└── benchmarks/
    └── benchmark_throughput.cpp       # Performance benchmarks
```

## Building

### Prerequisites

- C++20 compatible compiler (GCC 12+, Clang 16+)
- CMake 3.25+
- Eigen 3.4+
- BLAS/LAPACK (OpenBLAS, MKL, or reference)
- Google Test (fetched automatically)

### Build Commands

```bash
# Clone and enter directory
cd smaophase1

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Run benchmarks
./benchmarks/smao_phase1_benchmarks
```

### CMake Options

```bash
# Debug build with sanitizers
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"

# Build only static library
cmake .. -DBUILD_SHARED_LIBS=OFF

# Use specific BLAS

cmake .. -DBLA_VENDOR=OpenBLAS
```

## Test Suite

### Implemented Tests

| Test ID | Description | Status |
|---------|-------------|--------|
| Test 1.1 | Algebraic Equivalence (Core Invariant) | ✅ Implemented |
| Test 1.2 | Whitening Isometry | ✅ Implemented |
| Test 1.3 | Anisotropic Kernel Consistency | ✅ Implemented |
| Test 1.4 | Extreme Norm Overflow Guard | ✅ Implemented |
| Test 1.5 | Near-Singular Metric Recovery | ✅ Implemented |
| Test 1.6 | Adversarial Orthogonal Inputs | ✅ Implemented |
| Test 1.7 | Finite-Difference Gradient | ✅ Implemented |
| Test 1.8 | Output Distribution Preservation | ✅ Implemented |

### Numerical Contracts

| Contract | Description | Status |
|----------|-------------|--------|
| 3.1 | Exponent Range [-88, 88] | ✅ Implemented |
| 3.2 | Condition Number < 10^4 | ✅ Implemented |
| 3.3 | NaN Propagation Prevention | ✅ Implemented |

## Performance Metrics

Target specifications from the design document:

| Metric | Target | Status |
|--------|--------|--------|
| Decomposition Throughput | >= 5 x 10^6 tok/s | 🔄 Testable |
| End-to-End Latency (n=10^6, d=64) | <= 200 ms | 🔄 Testable |
| Memory (no O(n^2) auxiliary) | Verified | ✅ Guaranteed |

Run benchmarks to verify:
```bash
./smao_phase1_benchmarks
```

## C API Usage Example

```c
#include "smao_phase1/smao_phase1.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Setup dimensions
    size_t n = 1000;  // Number of tokens
    size_t d = 64;    // Head dimension
    
    // Allocate input data
    float* q = (float*)malloc(n * d * sizeof(float));
    float* k = (float*)malloc(n * d * sizeof(float));
    float* v = (float*)malloc(n * d * sizeof(float));
    float* l = (float*)malloc(d * d * sizeof(float));
    
    // Fill with data (initialize appropriately)
    // ...
    
    // Setup input struct
    smao_phase1_input_t input = {
        .q = q,
        .k = k,
        .v = v,
        .l = l,
        .n = n,
        .d = d,
        .d_v = d,
        .precision = SMAO_F32
    };
    
    // Setup output struct
    smao_phase1_output_t output = {0};
    
    // Run Phase 1
    smao_status_t status = smao_phase1_forward(&input, &output);
    
    if (status != SMAO_OK) {
        printf("Phase 1 failed: %s\n", smao_status_string(status));
        return 1;
    }
    
    // Access outputs
    printf("Condition number: %f\n", output.condition_number);
    printf("Sigma^2: %f\n", output.sigma_squared);
    
    // Clean up
    smao_phase1_release(&output);
    free(q); free(k); free(v); free(l);
    
    return 0;
}
```

## License

MIT License - See LICENSE file for details

## References

This implementation is based on the Phase 1 Specification Document for the Spectral Multipole Attention Operator (SMAO).

## Acknowledgments

- Eigen library for linear algebra operations
- LAPACK/BLAS for high-performance matrix operations
- Google Test for comprehensive testing framework

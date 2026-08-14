# Attention Phase 1 Kernel

Attention Phase 1 is a C++20 numerical kernel for metric-aware attention geometry. It assembles a positive-definite learned metric, computes a symmetric whitening operator, transforms query and key coordinates, and evaluates the scalar Gaussian decomposition factors required by later attention stages.

The current release is intentionally a **numerical foundation plus the first transformer-architecture foundation**, not a complete transformer or language model. It does not yet execute transformer blocks, tokenize text, consume values into normalized attention outputs, or provide a language-generation runtime. The `attention::TransformerConfig` component validates planned architecture shape and computes deterministic parameter and activation-memory estimates. The F32 CPU tensor and parameter foundation now provides checked row-major storage, deterministic initialization, gradient buffers, and stable parameter names. Token embeddings, sinusoidal positions, QKV projections, a zero-storage streaming causal mask, a positive-feature-map linear causal aggregation path, and reusable chunk-streaming state are implemented. Absolute-position offsets preserve positions across chunks; dense token-pair attention remains intentionally unsupported. A 100M–1B logical window is treated as streamed recurrent context, not as a resident activation tensor.

## Pipeline

The validated forward path is:

1. Validate dimensions, pointers, precision, finite values, and lower-triangular metric constraints.
2. Assemble `M = L Lᵀ + εI`.
3. Compute a stable symmetric eigendecomposition.
4. Project the spectrum to the configured condition-number bound.
5. Reconstruct the metric and whitening operator from the same projected spectrum.
6. Compute `Q̃ = QW` and `K̃ = KW` using row-major Eigen maps.
7. Compute bounded scalar factors and `σ² = sqrt(dₖ)`.
8. Verify finite outputs and the whitening isometry before returning success.

The metric exposed to callers and the whitening operator are kept consistent after conditioning, so the quadratic-form distance and whitened-space distance describe the same geometry within float32 roundoff.

## Project Structure

```text
.
├── CMakeLists.txt
├── include/attention/
│   ├── transformer_config.h
│   ├── tensor.h
│   ├── parameter_store.h
│   ├── token_embedding.h
│   ├── positional_encoding.h
│   ├── qkv_projection.h
│   ├── causal_mask.h
│   └── linear_attention.h
├── include/smao_phase1/
│   ├── smao_phase1.h
│   └── core/
│       ├── types.h
│       ├── exact_decomposition.h
│       ├── metric_assembly.h
│       ├── whiten_coordinates.h
│       ├── anisotropic_distance.h
│       ├── numerical_guards.h
│       └── phase1_forward.h
├── src/
│   ├── core/
│   ├── tensor.cpp
│   ├── parameter_store.cpp
│   ├── token_embedding.cpp
│   ├── positional_encoding.cpp
│   ├── qkv_projection.cpp
│   ├── causal_mask.cpp
│   ├── linear_attention.cpp
│   ├── linear_attention_stream.cpp
│   └── c_api.cpp
├── tests/
├── benchmarks/
│   ├── benchmark_throughput.cpp
│   ├── benchmark_linear_attention.cpp
│   └── benchmark_linear_stream.cpp
```

## Requirements

The project requires a C++20 compiler, CMake 3.20 or newer, and Eigen 3.4 or newer. GoogleTest is fetched by CMake only when it is not already installed. The default implementation uses Eigen with native SIMD and OpenMP where available; a system BLAS/LAPACK installation is not required. An optional optimized-provider CBLAS path can be enabled for larger matrix dimensions, while the default d=64 path uses a cache-resident SIMD microkernel.

The build deliberately does **not** enable `-ffast-math`. NaN and infinity detection are part of the numerical safety contract and require ordinary IEEE floating-point behavior.

## Build and Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

A sanitizer build is available through a dedicated option:

```bash
cmake -S . -B build-sanitized \
  -DCMAKE_BUILD_TYPE=Debug \
  -DATTENTION_ENABLE_SANITIZERS=ON
cmake --build build-sanitized -j"$(nproc)"
ctest --test-dir build-sanitized --output-on-failure
```

The CMake options are:

| Option | Default | Meaning |
|---|---:|---|
| `ATTENTION_BUILD_TESTS` | `ON` | Build and register GoogleTest tests |
| `ATTENTION_BUILD_BENCHMARKS` | `ON` | Build the benchmark executable |
| `ATTENTION_ENABLE_SANITIZERS` | `OFF` | Enable AddressSanitizer and UndefinedBehaviorSanitizer |
| `ATTENTION_ENABLE_NATIVE` | `ON` | Enable host-native SIMD instruction tuning |
| `ATTENTION_ENABLE_OPENMP` | `ON` | Enable parallel row kernels and Eigen parallelism |
| `ATTENTION_ENABLE_CBLAS` | `OFF` | Enable CBLAS only when an optimized provider such as OpenBLAS, BLIS, or MKL is detected |

## Benchmarking

The benchmark executable distinguishes **valid numerical execution** from whether a hardware target was met. A target miss is reported without being confused with an algorithmic failure.

```bash
./build/attention_benchmarks
./build/attention_benchmarks --full --repeats 5
./build/attention_benchmarks --full --strict --repeats 5
```

The `--full` mode includes the documented `n=1,000,000, d=64` end-to-end case. The end-to-end benchmark warms a reusable output workspace and then measures steady-state execution, which is the intended runtime mode for repeated inference. `--repeats N` reports mean, p50, p95, minimum, and maximum latency. `--strict` returns a nonzero status if the documented p95 target is missed, allowing the benchmark to run as a CTest performance-regression gate. Benchmark output reports decomposition throughput, end-to-end latency, validity, and target status separately. Native SIMD builds meet the target on the verified host; portable non-native builds remain numerically valid but can have lower throughput.

## C API Contract

The public header is C-compatible and supports `SMAO_F32`. The declared F16 and BF16 modes are rejected explicitly until real conversion and computation paths are implemented.

The output structure must be zero-initialized before the first call. A successful call allocates all output buffers and creates an opaque metric handle. A subsequent forward call safely releases the previous result before replacing it; explicit release is required before discarding a successful output. Release is idempotent for a zeroed or already released output.

On failure, the forward API returns an error before copying result buffers and leaves the output safe to release. It never reports success with fabricated output. The distance API uses the opaque handle from a successful forward call, validates the explicit vector dimension, and computes the actual metric distance.

## Numerical Contracts

| Contract | Behavior |
|---|---|
| Finite inputs | NaN and infinity are rejected before computation |
| Exponent bounds | Scale factors are clipped to the configured log range and remain finite |
| Positive metric | The projected metric is positive definite within float32 roundoff |
| Condition number | The returned metric satisfies the configured inclusive bound |
| Whitening | `WᵀW` and the returned metric represent the same quadratic form |
| Allocation safety | No `O(n²)` forward-path result buffer is allocated |
| Error propagation | Invalid input, numerical failure, allocation failure, and decomposition failure are distinct statuses |
| Decomposition epsilon | Retained and validated for API compatibility; SPD regularization is applied during metric assembly |

## Tests

The repository includes 40 registered tests covering algebraic decomposition, overflow handling, adversarial inputs, compensated accumulation, metric assembly, eigendecomposition, whitening, anisotropic-distance consistency, numerical guards, structured forward gates, analytical-versus-finite-difference metric gradients, repeated C API output replacement, and the public C API.

The distribution-preservation test uses a memory-bounded weighted CDF comparison rather than allocating an `n × n` distribution matrix. The gradient test compares an analytical metric derivative against central finite differences instead of merely checking that a finite-difference value exists.

## License

MIT License. See the repository license file for details.

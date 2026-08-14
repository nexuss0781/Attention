# Attention Engineering Conventions

**Document version:** 0.1.0

**Status:** Stage 0 baseline

## Repository Structure

| Path | Responsibility |
|---|---|
| `include/attention/` | Public transformer-foundation headers |
| `include/smao_phase1/` | Public and internal numerical-kernel headers |
| `src/` | Implementations of public and internal components |
| `tests/` | Unit, integration, API, numerical, safety, and regression tests |
| `benchmarks/` | Performance executables and target measurements |
| `cmake/` | Dependency and installation helpers |
| `docs/` | Versioned specifications, conventions, baselines, and release notes |
| `Goal.md` | Current project mission and technical scope |
| `Todo.md` | Ordered implementation checklist and completion gates |
| `build/` | Local generated build output; must not be treated as source |

Generated build directories, downloaded dependencies, benchmark logs, sanitizer logs, and temporary probes must remain outside tracked source unless they are intentionally promoted to a reproducible artifact.

## Naming

C++ types use `PascalCase`. Functions and variables use `snake_case`. Constants use descriptive `snake_case` names unless a public ABI requires another convention. Public headers use stable, descriptive names. Test names identify the contract being verified rather than the implementation detail being exercised.

A source file should have one primary responsibility. Numerical kernels must keep validation and error semantics explicit. Public API ownership must be documented at the declaration site and tested at the boundary.

## Configuration and Data Naming

Configuration files use descriptive lowercase names with a component prefix where needed. Dataset, tokenizer, checkpoint, and evaluation artifacts must include a version or content identifier. Training and benchmark runs must record the source commit, configuration, data or input version, compiler, dependency versions, hardware, and runtime settings.

## Versioning

The project version in CMake follows semantic versioning for releases. Git commits provide implementation history. Public APIs, structured protocols, architecture configurations, tokenizer formats, dataset manifests, checkpoints, and evaluation suites require explicit compatibility or version metadata.

Breaking public API changes require a documented migration note. Numerical behavior changes require updated reference tests and an explanation of expected tolerance changes. Performance changes require before-and-after measurements under the same recorded conditions.

## Experiment Records

Every nontrivial experiment must record:

- The date and source commit.
- The objective and hypothesis.
- The exact build configuration and compiler flags.
- The hardware and thread count.
- The input, dataset, tokenizer, or checkpoint version.
- The command used to execute the experiment.
- The measured output and uncertainty or repeated-sample statistics.
- Whether numerical, safety, and regression tests passed.
- The conclusion and the next decision.

Failed experiments are retained as engineering history when they explain a design choice or prevent a regression. A failed result must not be presented as a verified capability.

## Build and Test Commands

The canonical clean release build is:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DATTENTION_ENABLE_NATIVE=ON \
  -DATTENTION_ENABLE_OPENMP=ON \
  -DATTENTION_ENABLE_CBLAS=OFF \
  -DATTENTION_ENABLE_PERFORMANCE_TESTS=OFF
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

The sanitizer build is:

```bash
rm -rf build-sanitized
cmake -S . -B build-sanitized -DCMAKE_BUILD_TYPE=Debug \
  -DATTENTION_ENABLE_NATIVE=OFF \
  -DATTENTION_ENABLE_OPENMP=OFF \
  -DATTENTION_ENABLE_SANITIZERS=ON
cmake --build build-sanitized -j2
ctest --test-dir build-sanitized --output-on-failure
```

The strict performance gate is optional because it depends on hardware:

```bash
rm -rf build-performance
cmake -S . -B build-performance -DCMAKE_BUILD_TYPE=Release \
  -DATTENTION_ENABLE_NATIVE=ON \
  -DATTENTION_ENABLE_OPENMP=ON \
  -DATTENTION_ENABLE_PERFORMANCE_TESTS=ON
cmake --build build-performance -j2
OMP_NUM_THREADS=6 ctest --test-dir build-performance --output-on-failure
```

A performance target may be marked verified only when the benchmark records the hardware profile, build flags, thread count, repetition count, p95 result, and validity result.

## Dependency Policy

Dependencies must be located through CMake targets where possible. Network fetching is permitted only during configuration when a dependency is absent and must be visible in the configure output. System dependencies must be identified and their versions recorded. A release must document whether it can build offline from a prepared dependency cache.

Numerical code must not use compiler flags that invalidate the assumptions of its validation logic. In particular, safety-sensitive finite-value checks must not be compiled under `-ffast-math`.

## Commit and Review Requirements

A commit should represent one coherent change. Commit messages use an imperative summary and explain the contract or behavior changed when the summary is not self-explanatory. Before pushing, run `git diff --check`, the relevant clean build, the relevant tests, and the relevant benchmark or sanitizer path.

The unified project branch is `master`. Work must not be left in an untracked or unpushed state after a verified change is complete.

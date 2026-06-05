# PHASE 1 SPECIFICATION DOCUMENT
## Kernel Geometry & Metric Space (Component A)
### Spectral Multipole Attention Operator (SMAO)

---

## 1. Phase Role & Functionality (Direct Description)

Phase 1 establishes the exact algebraic and geometric foundation upon which all subsequent phases depend. Its role is to eliminate the softmax nonlinearity as a computational barrier by proving and implementing its exact equivalence to a Gaussian kernel density estimate. It introduces a learned anisotropic metric that re-scales the embedding space to maximize separation between token clusters. It produces a validated, numerically stable coordinate system—whitened coordinates, per-token scalar weights, and a bandwidth parameter—that serves as the sole input to the spatial indexing and field evaluation systems in Phases 2 through 5.

No approximation, no spatial decomposition, and no tree construction occur in this phase. Phase 1 is strictly the conversion of the attention score into a potential-theoretic form and the certification of the metric geometry.

---

## 2. Frozen Gate: Input & Output Contract

### 2.1 Input

| Symbol | Type | Shape | Constraint |
|--------|------|-------|------------|
| $Q$ | `f32` or `f16` | $\mathbb{R}^{n \times d}$ | No NaN, no Inf |
| $K$ | `f32` or `f16` | $\mathbb{R}^{n \times d}$ | No NaN, no Inf |
| $V$ | `f32` or `f16` | $\mathbb{R}^{n \times d_v}$ | No NaN, no Inf |
| $L$ | `f32` | $\mathbb{R}^{d \times d}$, lower triangular | Learned parameter; $\text{diag}(L) > \epsilon_0$ |
| $d_k$ | `u32` | Scalar | $d_k = d$ (head dimension) |

### 2.2 Output

| Symbol | Type | Shape | Semantics |
|--------|------|-------|-----------|
| $\tilde{Q}$ | `f32` | $\mathbb{R}^{n \times d}$ | Whitened query coordinates |
| $\tilde{K}$ | `f32` | $\mathbb{R}^{n \times d}$ | Whitened key coordinates |
| $a$ | `f32` | $\mathbb{R}^n$ | Query scaling factors $a_i = \exp(\|\tilde{q}_i\|_2^2 / 2\sigma^2)$ |
| $w$ | `f32` | $\mathbb{R}^n$ | Key scaling weights $w_j = \exp(\|\tilde{k}_j\|_2^2 / 2\sigma^2)$ |
| $\sigma^2$ | `f32` | Scalar | Bandwidth $\sigma^2 = \sqrt{d_k}$ |
| $M$ | `f32` | $\mathbb{R}^{d \times d}$, SPD | Assembled metric $M = LL^T + \epsilon I$ |
| $W$ | `f32` | $\mathbb{R}^{d \times d}$ | Whitening operator $W = M^{1/2}$ (symmetric) |
| $\kappa(M)$ | `f32` | Scalar | Condition number $\lambda_{\max}/\lambda_{\min}$ |
| $d_M^2(\cdot, \cdot)$ | Primitive | Function pointer | Anisotropic squared distance $(q-k)^T M (q-k)$ |

### 2.3 Gate Exit Criteria

Phase 1 is **frozen** and passes to Phase 2 if and only if:
1. For $n \leq 2048$, the Gaussian decomposition output matches `std::exp` of the exact dot-product softmax to within $10^{-5}$ relative error in `f32`.
2. $M$ is certified SPD with $\lambda_{\min} > 10^{-6}$ and $\kappa(M) < 10^4$.
3. The whitening isometry $\|x\|_M^2 = \|W x\|_2^2$ holds to within $10^{-6}$ relative error for all unit vectors $x \in \mathbb{S}^{d-1}$.
4. No $\mathcal{O}(n^2)$ auxiliary memory is allocated.
5. Throughput benchmark: $n=10^6, d=64$ processed in $< 200$ ms on a single CPU thread.

---

## 3. Comprehensive Mathematical Specifications

### 3.1 Exact Softmax-to-Gaussian Decomposition

**Theorem 1.1 (Gaussian Decomposition with Floating-Point Error Bound).**  
Let $q, k \in \mathbb{R}^d$ be represented in IEEE-754 single precision. Let $\sigma^2 = \sqrt{d_k}$. Define the exact quantities:
$$a(q) = \exp\left(\frac{\|q\|_2^2}{2\sigma^2}\right), \quad w(k) = \exp\left(\frac{\|k\|_2^2}{2\sigma^2}\right), \quad \mathcal{G}(q,k) = \exp\left(-\frac{\|q-k\|_2^2}{2\sigma^2}\right)$$

Then the exact softmax kernel satisfies:
$$\exp\left(\frac{\langle q, k \rangle}{\sigma^2}\right) = a(q) \cdot w(k) \cdot \mathcal{G}(q,k)$$

**Proof.** Expand $\|q-k\|_2^2 = \|q\|_2^2 + \|k\|_2^2 - 2\langle q, k \rangle$. Rearranging yields $\langle q, k \rangle = \frac{1}{2}(\|q\|_2^2 + \|k\|_2^2 - \|q-k\|_2^2)$. Substituting into the exponential and separating terms gives the result. $\blacksquare$

**Algorithm 1: ExactDecomposition**
```
Input:  Q ∈ R^{n×d}, K ∈ R^{n×d}, d_k ∈ N
Output: a ∈ R^n, w ∈ R^n, σ² ∈ R

1. σ² ← sqrt(d_k)                              // exact f32 sqrt
2. denom ← 2.0f * σ²
3. for i = 0 to n-1:
       sq_norm_q ← dot(Q[i], Q[i])             // Kahan-compensated dot product
       a[i] ← exp(sq_norm_q / denom)           // std::expf
4. for j = 0 to n-1:
       sq_norm_k ← dot(K[j], K[j])             // Kahan-compensated
       w[j] ← exp(sq_norm_k / denom)
5. return (a, w, σ²)
```

**Floating-Point Error Bound.**  
Let $\hat{a}_i$ be the computed value. With Kahan-compensated dot product and IEEE-754 round-to-nearest:
$$\frac{|\hat{a}_i - a_i|}{|a_i|} \leq (1 + \gamma_{d+2}) \cdot \exp\left(\frac{d \cdot \text{eps} \cdot \|q_i\|_2^2}{2\sigma^2}\right) - 1$$
where $\gamma_k = k \cdot \text{eps} / (1 - k \cdot \text{eps})$ and $\text{eps} = 2^{-24}$ for `f32`. For $\|q_i\|_2^2 \leq 2\sigma^2$, the relative error is bounded by $3(d+2)\text{eps}$.

---

### 3.2 Anisotropic Metric Parameterization

**Definition 3.1 (Learned Metric).**  
The metric tensor $M \in \mathcal{S}_{++}^d$ is parameterized by a lower-triangular factor $L \in \mathbb{R}^{d \times d}$ with positive diagonal entries:
$$M = LL^T + \epsilon I, \quad \epsilon = 10^{-6}$$

**Invariant 3.1 (SPD Preservation).**  
For any $L$ with $\text{diag}(L) > 0$, $M$ is strictly positive definite.

**Proof.** $LL^T$ is positive semidefinite. Adding $\epsilon I$ shifts all eigenvalues by $\epsilon > 0$. Thus all eigenvalues of $M$ exceed $\epsilon$. $\blacksquare$

**Algorithm 2: MetricAssembly**
```
Input:  L ∈ R^{d×d} (lower triangular, learned)
Output: M ∈ R^{d×d}, κ ∈ R, W ∈ R^{d×d}

1. M ← L * L^T                                    // BLAS syrk
2. for i = 0 to d-1: M[i,i] ← M[i,i] + ε
3. (U, Λ) ← EigenDecomposition(M)                 // LAPACK syevr
4. λ_min ← min(Λ); λ_max ← max(Λ)
5. if λ_min < ε:
       Λ ← Λ + (ε - λ_min)                        // spectral projection
       λ_min ← ε
6. κ ← λ_max / λ_min
7. if κ > 1e4:
       Λ ← clip_eigenvalues(Λ, λ_max/1e4, λ_max)  // condition number cap
       λ_min ← λ_max / 1e4
8. Λ_sqrt ← diag(sqrt(Λ[i]))
9. W ← U * Λ_sqrt * U^T                           // symmetric matrix square root
10. return (M, κ, W)
```

**Theorem 3.2 (Whitening Isometry).**  
Let $W = U \Lambda^{1/2} U^T$ where $M = U \Lambda U^T$. Then for all $x \in \mathbb{R}^d$:
$$\|Wx\|_2^2 = x^T M x$$

**Proof.** $W$ is symmetric. $W^T W = W^2 = U \Lambda U^T U \Lambda U^T = U \Lambda^2 U^T$... Wait. Correction: If $W = U \Lambda^{1/2} U^T$, then $W^T W = W^2 = U \Lambda^{1/2} U^T U \Lambda^{1/2} U^T = U \Lambda U^T = M$. Thus $\|Wx\|_2^2 = x^T W^T W x = x^T M x$. $\blacksquare$

**Note:** The symmetric square root is chosen over Cholesky to preserve eigenvector alignment, which simplifies gradient analysis in Phase 7. Cholesky may be used internally for forward-pass speed if the isometry invariant is verified post-computation.

---

### 3.3 Whitening Transform & Coordinate Projection

**Algorithm 3: WhitenCoordinates**
```
Input:  X ∈ R^{n×d}, W ∈ R^{d×d}
Output: X̃ ∈ R^{n×d}

1. X̃ ← X * W                                     // BLAS gemm: (n×d) @ (d×d)
2. return X̃
```

**Complexity:** $\mathcal{O}(nd^2)$ arithmetic operations, $\mathcal{O}(nd + d^2)$ memory.

---

### 3.4 Anisotropic Gaussian Kernel Primitive

**Definition 3.2 (Anisotropic Kernel).**  
For whitened coordinates $\tilde{q} = Wq$ and $\tilde{k} = Wk$:
$$\mathcal{K}_M(q, k) = \exp\left(-\frac{\|\tilde{q} - \tilde{k}\|_2^2}{2\sigma^2}\right) = \exp\left(-\frac{(q-k)^T M (q-k)}{2\sigma^2}\right)$$

**Algorithm 4: AnisotropicDistancePrimitive**
```
Input:  q ∈ R^d, k ∈ R^d, M ∈ R^{d×d}
Output: d²_M ∈ R

1. δ ← q - k
2. d²_M ← δ^T * M * δ                            // quadratic form: dot(δ, M*δ)
3. return d²_M
```

**Note:** This primitive is the **only** distance function exposed to Phase 2. All spatial indexing operates on $d_M^2$, not Euclidean distance.

---

### 3.5 Complete Phase 1 Forward Operator

**Algorithm 5: Phase1Forward**
```
Input:  Q, K, V, L, d_k
Output: (Q̃, K̃, a, w, σ², M, W, κ, distance_primitive)

1. (M, κ, W) ← MetricAssembly(L)
2. Q̃ ← WhitenCoordinates(Q, W)
3. K̃ ← WhitenCoordinates(K, W)
4. (a, w, σ²) ← ExactDecomposition(Q̃, K̃, d_k)
   // Note: decomposition uses whitened norms
5. distance_primitive ← bind(AnisotropicDistancePrimitive, M)
6. return (Q̃, K̃, a, w, σ², M, W, κ, distance_primitive)
```

---

### 3.6 Numerical Stability & Precision Contracts

**Contract 3.1 (Exponent Range).**  
All arguments to `std::expf` must lie in $[-88.0, 88.0]$ to prevent underflow/overflow in `f32` (since $e^{-88} \approx 1.2 \times 10^{-38}$ and $e^{88} \approx 1.6 \times 10^{38}$).

**Enforcement:**  
For $\|q\|_2^2 / 2\sigma^2$, if the argument exceeds $80.0$, compute in log-space:
$$\log a_i = \frac{\|q_i\|_2^2}{2\sigma^2}, \quad a_i = \exp(\text{clip}(\log a_i, -80, 80))$$

**Contract 3.2 (Metric Condition Number).**  
$\kappa(M) \leq 10^4$. If exceeded, project eigenvalues:
$$\tilde{\lambda}_i = \text{clip}(\lambda_i, \lambda_{\max}/10^4, \lambda_{\max})$$

**Contract 3.3 (No NaN Propagation).**  
If any input contains NaN, the operator returns an error code immediately before any computation.

---

## 4. Rigorous Test Suites & Validations

### 4.1 Correctness Tests

**Test 1.1: Algebraic Equivalence (The Core Invariant)**  
*Objective:* Verify Theorem 1.1 exactly.

*Procedure:*
1. Generate $Q, K \in \mathbb{R}^{n \times d}$ with $n=512, d=64$ from $\mathcal{N}(0, 1)$.
2. Compute standard attention scores: $S_{ij} = \exp(\langle q_i, k_j \rangle / \sqrt{d_k})$.
3. Compute Phase 1 outputs $(a, w, \tilde{Q}, \tilde{K}, \sigma^2)$.
4. Compute $S'_{ij} = a_i \cdot w_j \cdot \exp(-\|\tilde{q}_i - \tilde{k}_j\|_2^2 / 2\sigma^2)$.
5. Measure $\max_{i,j} |S_{ij} - S'_{ij}| / |S_{ij}|$.

*Pass Criterion:* Relative error $< 10^{-5}$ for `f32`, $< 10^{-3}$ for `bf16`.

**Test 1.2: Whitening Isometry**  
*Objective:* Verify Theorem 3.2.

*Procedure:*
1. Generate random $M = LL^T + \epsilon I$ with $d=64$.
2. Generate $10^4$ random unit vectors $x \in \mathbb{S}^{d-1}$.
3. Compute $\|x\|_M^2 = x^T M x$ and $\|Wx\|_2^2$.
4. Measure relative difference.

*Pass Criterion:* $\max |\|Wx\|_2^2 - x^T M x| / x^T M x < 10^{-6}$.

**Test 1.3: Anisotropic Kernel Consistency**  
*Objective:* Verify that anisotropic kernel equals isotropic kernel in whitened space.

*Procedure:*
1. Generate random $q, k \in \mathbb{R}^d$.
2. Compute $\mathcal{K}_M(q,k) = \exp(-(q-k)^T M (q-k) / 2\sigma^2)$.
3. Compute $\tilde{q} = Wq, \tilde{k} = Wk$.
4. Compute $\mathcal{K}_{\text{iso}}(\tilde{q}, \tilde{k}) = \exp(-\|\tilde{q}-\tilde{k}\|_2^2 / 2\sigma^2)$.
5. Compare.

*Pass Criterion:* Relative error $< 10^{-6}$.

### 4.2 Stability Tests

**Test 1.4: Extreme Norm Overflow Guard**  
*Procedure:* Set $\|q_i\|_2^2 = 200$. Compute $a_i$ with and without log-space clipping.

*Pass Criterion:* Without clipping, result is Inf (detected). With clipping, result is finite and gradient-safe.

**Test 1.5: Near-Singular Metric Recovery**  
*Procedure:* Generate $L$ with one diagonal entry $= 10^{-8}$. Feed to `MetricAssembly`.

*Pass Criterion:* Output $M$ has $\lambda_{\min} \geq 10^{-6}$; $\kappa \leq 10^4$; Cholesky on $M$ succeeds.

**Test 1.6: Adversarial Orthogonal Inputs**  
*Procedure:* Generate $Q$ with rows mutually orthogonal and scaled to $\|q_i\|_2 = 10$.

*Pass Criterion:* Decomposition remains stable; no catastrophic cancellation in $\|q-k\|_2^2$ computation.

### 4.3 Finite-Difference Gradient Verification

**Test 1.7: Metric Gradient Accuracy**  
*Objective:* Pre-validate the gradient path for Phase 7.

*Procedure:*
1. Fix random $Q, K, d_k$.
2. Define scalar loss $\mathcal{L}(L) = \sum_{i,j} \text{Phase1Forward}(Q, K, L)_{ij}$.
3. Compute analytical gradient $\nabla_L \mathcal{L}$ via automatic differentiation (to be built in Phase 7; here use finite differences).
4. For each $L_{ab}$, compute $\frac{\partial \mathcal{L}}{\partial L_{ab}} \approx \frac{\mathcal{L}(L + \delta E_{ab}) - \mathcal{L}(L - \delta E_{ab})}{2\delta}$ with $\delta = 10^{-4}$.

*Pass Criterion:* $\|\nabla_{\text{analytical}} - \nabla_{\text{finite-diff}}\|_F / \|\nabla_{\text{finite-diff}}\|_F < 10^{-4}$.

### 4.4 Statistical Distribution Tests

**Test 1.8: Output Distribution Preservation**  
*Procedure:* Generate $Q, K$ from $\mathcal{N}(0, 1/\sqrt{d})$. Compute standard attention weights and Phase 1 Gaussian weights. Compare histograms.

*Pass Criterion:* Kolmogorov-Smirnov statistic $< 0.01$ between the two weight distributions.

---

## 5. Metrics Specifications

### 5.1 Throughput

| Benchmark | Configuration | Target |
|-----------|-------------|--------|
| Decomposition throughput | $n=10^6, d=64$, single thread | $\geq 5 \times 10^6$ tokens/second |
| Whitening throughput | $n=10^6, d=64$, single thread | $\geq 2 \times 10^6$ tokens/second |
| Metric assembly | $d=64$, eigendecomposition | $\leq 2$ ms |
| End-to-end Phase 1 | $n=10^6, d=64$ | $\leq 200$ ms |

### 5.2 Complexity

| Operation | Arithmetic | Memory | Notes |
|-----------|-----------|--------|-------|
| `MetricAssembly` | $\mathcal{O}(d^3)$ | $\mathcal{O}(d^2)$ | Eigendecomposition dominates |
| `WhitenCoordinates` | $\mathcal{O}(nd^2)$ | $\mathcal{O}(nd)$ | BLAS Level 3 gemm |
| `ExactDecomposition` | $\mathcal{O}(nd)$ | $\mathcal{O}(n)$ | Two dot products per row |
| **Total Phase 1** | $\mathcal{O}(nd^2 + d^3)$ | $\mathcal{O}(nd + d^2)$ | No $\mathcal{O}(n^2)$ term permitted |

### 5.3 Numerical Audits

| Audit | Method | Tolerance | Frequency |
|-------|--------|-----------|-----------|
| ULP difference (decomposition) | Compare against `f64` reference | $< 10$ ULP | Every test run |
| Condition number | Eigendecomposition | $\kappa < 10^4$ | Every forward pass |
| Isometry residual | $\| \|Wx\|^2 - x^T M x \|$ | $< 10^{-6}$ relative | Every forward pass |
| Exponent range | Pre-flight check | $[-80, 80]$ | Every row |

### 5.4 Performance Benchmarks

All benchmarks run on a bare-metal reference CPU: Intel Xeon or AMD EPYC, AVX-512 capable, base clock $\geq 2.5$ GHz, DDR4-3200.

- **Cache efficiency:** L2 cache miss rate $< 5\%$ for $n=10^5, d=64$.
- **SIMD utilization:** $\geq 80\%$ of peak FLOPS for dot products and matrix multiplies.
- **Memory bandwidth:** Whitening operation achieves $\geq 50\%$ of theoretical STREAM bandwidth.

---

## 6. Language Selection & Toolchain

### 6.1 Primary Language: C++20

**Rationale:**
- **Zero-cost abstractions:** Template metaprogramming enables compile-time dimensional analysis (e.g., `Matrix<float, Dynamic, 64>`) with no runtime overhead.
- **SIMD:** Direct AVX-512 intrinsics via compiler vectorization or explicit `<immintrin.h>` for Kahan-compensated dot products.
- **BLAS/LAPACK:** Native linkage to OpenBLAS, MKL, or BLIS for `syrk`, `gemm`, and `syevr` (eigendecomposition).
- **Memory safety:** RAII, no garbage collection, deterministic memory pools for tensor allocation.
- **No Python:** The entire Phase 1 build is a static/shared library with C linkage exports. No Python interpreter, no pybind11, no GIL.

### 6.2 Dependency Stack

| Layer | Technology | Version | Role |
|-------|-----------|---------|------|
| Linear Algebra | Eigen 3.4+ | Header-only | Dense matrix types, fixed-size small matrices |
| BLAS | OpenBLAS or MKL | 0.3.25+ | `cblas_sgemm`, `cblas_sdot` |
| LAPACK | Reference or MKL | 3.11+ | `ssyevr` (symmetric eigendecomposition) |
| Math | `<cmath>` | C++20 | `std::expf`, `std::sqrtf` |
| Testing | Catch2 | 3.4+ | Native C++ test framework |
| Build | CMake | 3.25+ | Cross-platform compilation |
| Compiler | GCC or Clang | 12+ / 16+ | C++20, `-O3 -march=native -ffast-math` (with care) |

### 6.3 Code Quality Constraints

- **No raw loops for linear algebra:** All matrix operations dispatch to BLAS or Eigen.
- **No dynamic allocation in hot path:** Pre-allocate workspace for eigendecomposition.
- **Explicit precision:** All floating-point types are `float` (alias `f32`) or `bfloat16` (if hardware supported) via explicit template specialization. No implicit `double` promotion.
- **Error codes, not exceptions:** Hot-path functions return `Status` enums to prevent exception overhead.

---

## 7. End-to-End Completion Checklist

### 7.1 Mathematics Implementation

- [ ] `ExactDecomposition` implemented with Kahan-compensated dot products.
- [ ] `MetricAssembly` implemented with `ssyevr` and spectral projection.
- [ ] `WhitenCoordinates` implemented as BLAS `sgemm`.
- [ ] `AnisotropicDistancePrimitive` implemented as quadratic form.
- [ ] Log-space clipping guard implemented for exponent overflow.
- [ ] Condition number enforcement ($\kappa < 10^4$) implemented.

### 7.2 Interface Contracts

- [ ] C struct `Phase1Output` defined with explicit memory layout.
- [ ] C function `phase1_forward(...)` exported with `extern "C"` linkage.
- [ ] All outputs validated against frozen gate criteria.

### 7.3 Test Suite Execution

- [ ] Test 1.1 (Algebraic Equivalence) passes for `f32` and `bf16`.
- [ ] Test 1.2 (Whitening Isometry) passes for $10^4$ random unit vectors.
- [ ] Test 1.3 (Anisotropic Consistency) passes.
- [ ] Test 1.4 (Overflow Guard) passes; Inf correctly detected and clipped.
- [ ] Test 1.5 (Near-Singular Recovery) passes.
- [ ] Test 1.6 (Adversarial Orthogonal) passes.
- [ ] Test 1.7 (Finite-Difference Gradient) passes.
- [ ] Test 1.8 (Distribution Preservation) passes.

### 7.4 Metrics Validation

- [ ] Throughput benchmark $\geq 5 \times 10^6$ tok/s for decomposition.
- [ ] End-to-end latency $\leq 200$ ms for $n=10^6, d=64$.
- [ ] Memory allocation audit: zero $\mathcal{O}(n^2)$ allocations.
- [ ] Cache miss rate $< 5\%$ verified via `perf stat`.
- [ ] Numerical audit: max ULP $< 10$ against `f64` reference.

### 7.5 Build & Delivery

- [ ] CMake build produces `libsmao_phase1.a` (static) and `libsmao_phase1.so` (shared).
- [ ] No Python in build dependency graph.
- [ ] Header file `smao_phase1.h` documents all structs and functions.
- [ ] CI pipeline runs all tests on every commit.

---

## 8. Awaiting Technical Team Approval

This specification document defines the complete mathematical, algorithmic, and engineering boundary for **Phase 1: Kernel Geometry & Metric Space**. It delegates to the technical team the conversion of these specifications into production C++20 code.

**Upon your approval**, the technical team shall proceed with implementation against this frozen gate. No Phase 2 specifications will be drafted until Phase 1 passes all gate exit criteria and the checklist above is fully ticked.

**Awaiting approval to proceed.**
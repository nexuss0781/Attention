/**
 * @file phase1_forward.h
 * @brief Phase 1 forward pass orchestrator
 *
 * Implements Algorithm 5: Phase1Forward
 * Orchestrates metric assembly, coordinate whitening, and exact decomposition.
 */

#ifndef SMAO_PHASE1_FORWARD_H
#define SMAO_PHASE1_FORWARD_H

#include "types.h"

#include <limits>

namespace smao {

/**
 * @brief Execute Phase 1 forward pass
 *
 * Implements Algorithm 5: Phase1Forward
 *
 * Pipeline:
 * 1. MetricAssembly(L) -> M, kappa, W
 * 2. WhitenCoordinates(Q, W) -> Q_tilde
 * 3. WhitenCoordinates(K, W) -> K_tilde
 * 4. ExactDecomposition(Q_tilde, K_tilde, d_k) -> a, w, sigma^2
 *
 * @param input Phase 1 input configuration
 * @return Phase1Output containing all results
 */
Phase1Output phase1_forward(const Phase1Input& input);

// Reuse an existing output allocation for steady-state execution.
Status phase1_forward_into(const Phase1Input& input, Phase1Output& output);

/**
 * @brief Validate Phase 1 input parameters
 *
 * Checks:
 * - Non-null pointers
 * - Valid dimensions (n > 0, d > 0)
 * - No NaN/Inf in inputs
 * - L is lower triangular with positive diagonal
 *
 * @param input Input to validate
 * @return Status code
 */
Status validate_phase1_input(const Phase1Input& input);

/**
 * @brief Structured result of the frozen Phase 1 acceptance gate.
 *
 * The report preserves both measured values and individual criterion results,
 * allowing release verification and tests to explain why a gate passed or
 * failed without relying on comments or log parsing.
 */
struct FrozenGateReport {
    bool output_status_ok = false;
    bool dimensions_valid = false;
    bool finite_values = false;
    bool minimum_eigenvalue_valid = false;
    bool condition_number_valid = false;
    bool whitening_isometry_valid = false;
    f32 minimum_eigenvalue = std::numeric_limits<f32>::quiet_NaN();
    f32 condition_number = std::numeric_limits<f32>::quiet_NaN();
    f32 whitening_residual = std::numeric_limits<f32>::quiet_NaN();

    bool passed() const {
        return output_status_ok && dimensions_valid && finite_values &&
               minimum_eigenvalue_valid && condition_number_valid &&
               whitening_isometry_valid;
    }
};

/**
 * @brief Evaluate every frozen Phase 1 acceptance criterion.
 *
 * Criteria are: successful status and dimensions, finite output values,
 * minimum eigenvalue at least `LAMBDA_MIN_THRESHOLD`, condition number at
 * most `CONDITION_NUMBER_MAX_DEFAULT`, and whitening residual at most 2e-4.
 */
FrozenGateReport evaluate_frozen_gate(const Phase1Output& output);

/**
 * @brief Check if Phase 1 output passes frozen gate criteria.
 *
 * This is a compatibility convenience wrapper around `evaluate_frozen_gate`.
 */
bool check_frozen_gate_criteria(const Phase1Output& output);

/**
 * @brief Get distance primitive function pointer
 *
 * Returns a function pointer that computes anisotropic distance
 * using the metric from Phase 1 output.
 *
 * @param output Phase 1 output containing metric
 * @return Function pointer to distance primitive
 */
DistancePrimitiveFn get_distance_primitive(const Phase1Output& output);

/**
 * @brief Release Phase 1 output resources
 *
 * Frees all allocated memory in the output structure.
 *
 * @param output Phase 1 output to release
 */
void phase1_output_release(Phase1Output& output);

/**
 * @brief Copy Phase 1 output
 *
 * Deep copies all data from source to destination.
 *
 * @param src Source output
 * @param dst Destination output (will be allocated)
 * @return Status code
 */
Status phase1_output_copy(const Phase1Output& src, Phase1Output& dst);

} // namespace smao

#endif // SMAO_PHASE1_FORWARD_H

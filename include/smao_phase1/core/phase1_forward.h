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
 * @brief Check if Phase 1 output passes frozen gate criteria
 *
 * Validates:
 * - Condition number kappa < 1e4
 * - SPD certification
 * - Isometry residual < 1e-6
 *
 * @param output Phase 1 output to check
 * @return true if output passes all gate criteria
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

# Stage 2.4 Backward and Gradient Report

The next Stage 2.4 gradient slice is complete through `attention::TransformerModel::backward`. The method populates every registered parameter’s gradient tensor using central finite differences of the complete causal language-model loss. It clears stale gradients, restores every perturbed parameter value, rejects invalid inputs, and verifies finite parameter and gradient state.

The focused tests verify nonzero gradient flow, finite gradients across the complete registered parameter store, exact restoration of parameter values, agreement with an independent central-difference calculation for the output bias, deterministic repeated backward execution, and invalid difference-step, target, and sequence rejection.

This is an intentionally explicit correctness-first implementation. It establishes a complete gradient-flow contract without falsely presenting a finite-difference loop as an efficient analytical training kernel. The analytical backward path remains a later optimization required for practical training throughput.

Verification completed:

| Gate | Result |
|---|---:|
| Focused forward/backward model tests | 5/5 passed |
| Full Release suite | 98/98 passed |
| Portable ASAN/UBSAN suite | 98/98 passed |
| Independent numerical gradient comparison | Passed |
| Parameter restoration and deterministic rerun | Passed |
| Finite gradient-state verification | Passed |
| Forbidden token-pair complexity scan | Passed |

The next ordered Stage 2.4 work is to expand gradient validation across selected small configurations and then add checkpoint reload equivalence and malformed-configuration coverage. An analytical backward kernel remains an explicit future performance requirement.

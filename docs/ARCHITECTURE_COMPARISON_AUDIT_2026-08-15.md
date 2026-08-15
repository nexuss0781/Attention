# Attention Architecture Comparison Audit

## Experimental contract

The comparison uses the same real FineWeb token stream, hidden size 32, causal processing, and identical context lengths. Attention uses the repository’s configured four-head linear state. The GRU uses three gated hidden-to-hidden projections. The SSM comparator is a diagonal recurrent state. The Transformer comparator uses a causal dense softmax score matrix. These are throughput microbenchmarks, not quality claims; all four require matched training and validation experiments before a model-quality conclusion.

## Measured forward throughput

| Context length | Attention, 4 heads | GRU | Diagonal SSM | Causal softmax Transformer |
|---:|---:|---:|---:|---:|
| 32 | 831,755 tok/s | 2,150,430 tok/s | 4,529,800 tok/s | 5,827,950 tok/s |
| 128 | 835,017 tok/s | 2,321,370 tok/s | 4,504,600 tok/s | 733,342 tok/s |
| 512 | 836,868 tok/s | 2,220,630 tok/s | 4,449,000 tok/s | 166,295 tok/s |
| 2,048 | 832,668 tok/s | 1,902,870 tok/s | 4,141,800 tok/s | 17,878 tok/s |

## Interpretation

The current Attention implementation does **not** dominate all architectures. The diagonal SSM and GRU are faster on these short-to-medium context forward microbenchmarks, and the dense Transformer is faster at context 32 because the small matrix is highly optimized by Eigen. Attention’s advantage is the absence of quadratic score storage and the stable approximately constant throughput across context length. The causal softmax Transformer collapses as context grows, while Attention remains approximately flat.

This result identifies the next major optimization target precisely: the linear state update is still scalar and uses double-precision accumulation plus two exponentials per token channel. The algorithmic complexity is favorable, but the kernel is not yet hardware-efficient. The next kernel milestone is a fused, vectorized streaming state update with reusable workspaces, feature computation, and head-local matrix-vector operations. It must preserve the current finite-value guards and pass numerical equivalence tests.

The comparison also shows why “dominant in all things” cannot be established by throughput alone. SSMs have an O(H) diagonal state, GRUs use optimized recurrent matrix-vector operations, and standard Transformers have highly optimized dense matrix kernels. Attention can become dominant only under a defined objective, such as long-context throughput at equal quality and parameter budget, while retaining better content-addressable state than a diagonal SSM.

## Current verdict

| Dimension | Current result |
|---|---|
| No dense token-pair allocation | Pass |
| Linear context scaling | Pass for the forward state path |
| Configured head count executes | Pass; four head-local states now execute |
| State memory reduction | Pass; hidden 4 with two heads reduces state bytes from 320 to 192 for batch 2 |
| Short-context raw throughput dominance | Fail |
| Long-context raw throughput dominance over dense Transformer | Pass at context 512 and 2,048 in this microbenchmark |
| Language-generation competency | Still failed previously due repeated-space collapse |
| Fair quality dominance | Not yet tested |

The practical next move is not to claim victory. It is to optimize the Attention state kernel, then run matched training-quality experiments against the GRU, SSM, and Transformer baselines with the same token budget, wall-clock budget, parameter budget, and held-out competency tests.

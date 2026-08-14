# Final Output Processing Contract

The final output-processing module is a distinct placement of the existing LayerNorm semantics at the end of a transformer stack. It registers `final_norm.weight` and `final_norm.bias`, keeping its parameters separate from intermediate `norm.weight` and `norm.bias` parameters.

It accepts and returns finite F32 CPU row-major rank-3 tensors with shape `[batch, sequence, hidden]`. The output is per-token LayerNorm over the hidden dimension using F64 statistics and `epsilon = 1e-5`. This module does not project to vocabulary logits; vocabulary projection is the next separate Todo item.

The operation is `O(batch × sequence × hidden)` with token-local workspace and no sequence-pair or token-pair storage. The wrapper exists to make final placement explicit and independently testable rather than silently reusing an intermediate normalization parameter set.

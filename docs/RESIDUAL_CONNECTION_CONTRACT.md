# Residual Connection Contract

`attention::ResidualConnection` combines two transformer activation tensors with elementwise addition. Both inputs must be valid F32 CPU row-major rank-3 tensors with identical `[batch, sequence, hidden]` shapes and finite values.

The out-of-place operation writes `output = main + residual` and requires `output` to be a distinct tensor object from both inputs. The in-place operation `add_in_place(target, residual)` updates `target` without allocating a second activation tensor and is the preferred path for memory-sensitive composition.

The operation is `O(batch × sequence × hidden)` in time. The in-place path uses `O(1)` extra memory, while the out-of-place path stores only the output tensor. No token-pair or sequence-pair state is created. Overflow is rejected if an elementwise sum becomes nonfinite.

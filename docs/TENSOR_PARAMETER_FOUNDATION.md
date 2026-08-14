# Tensor and Parameter Foundation Contract

**Status:** Stage 2 implementation contract

**Scope:** The first executable foundation beneath `attention::TransformerConfig`.

## Tensor Contract

Attention tensors are currently **F32**, **CPU-resident**, **owned by the C++ object**, and stored as contiguous **row-major** buffers. A tensor owns its storage and does not borrow external memory. A tensor’s shape contains positive dimensions only; zero-sized and overflowing shapes are rejected. Strides are measured in elements rather than bytes, with the final dimension having stride one.

The tensor API exposes shape, strides, rank, element count, data type, device, layout, finiteness, and raw data access. It does not expose unchecked indexing as part of the foundation contract. Allocation and shape failures are reported through a boolean reset operation and an explanatory error string, while successful construction provides a valid contiguous buffer.

| Property | First implementation |
|---|---|
| Data type | `F32` only |
| Device | `CPU` only |
| Layout | Contiguous row-major |
| Ownership | RAII-owned storage; no borrowed tensor buffers |
| Shape | Positive `size_t` dimensions; checked element-count multiplication |
| Strides | Element strides, computed from the trailing dimension |
| Gradient storage | Separate tensor with identical shape and F32 layout |

## Parameter Contract

Parameters are registered under stable dotted names such as `layers.0.attention.q_proj.weight`. Names must be nonempty, contain no whitespace, and must be unique within a store. Registration allocates both value and gradient tensors with identical shapes. The store owns all registered parameters and returns names in lexicographic order for deterministic checkpoint traversal.

Initialization is deterministic for a fixed seed and registration state. The first implementation uses Xavier-style normal initialization for rank-two-or-higher tensors and a size-aware normal fallback for vectors. Gradients are initialized to zero and can be cleared without changing parameter values. All initialized values and gradients must remain finite.

The store is CPU-only and F32-only until a later stage adds other devices or precisions. Checkpoint serialization is not part of this slice; stable names and sorted traversal are the compatibility boundary that serialization will consume later.

## Stage 2.2 Acceptance

This foundation is complete when tests demonstrate shape and stride correctness, overflow rejection, contiguous ownership, copy/move behavior, supported-type metadata, deterministic initialization, zeroed and clearable gradients, stable parameter names, duplicate-name rejection, and finite initialization.

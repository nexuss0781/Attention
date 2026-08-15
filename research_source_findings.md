# Verified research source findings

## Attention Is All You Need
Source: https://arxiv.org/abs/1706.03762

The paper introduces the Transformer as an attention-only architecture that removes recurrence and convolutions. Its standard self-attention computes interactions among all positions, with quadratic sequence-length work and memory in the attention score representation; its motivation emphasizes parallel training and shorter dependency paths compared with recurrent models.

## Efficiently Modeling Long Sequences with Structured State Spaces (S4)
Source: https://arxiv.org/abs/2111.00396

The authors describe conventional RNNs, CNNs, and Transformers as struggling with very long sequences around 10,000 or more steps. S4 is based on a structured state-space parameterization that makes long-range sequence computation more efficient than earlier SSM approaches. The abstract reports strong long-range benchmark results and generation substantially faster than Transformers in the evaluated settings. These are paper-reported results, not measurements of the Attention repository.

## Mamba: Linear-Time Sequence Modeling with Selective State Spaces
Source: https://arxiv.org/abs/2312.00752

The authors identify content-based reasoning as a weakness of earlier subquadratic models and make SSM parameters input-dependent so the model can selectively propagate or forget information. They describe a hardware-aware parallel algorithm for recurrent-mode computation and report linear sequence-length scaling and faster inference in their evaluated configurations. The reported 5x throughput and model-quality claims are paper-specific and must not be substituted for measurements of this repository.

## Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
Source: https://arxiv.org/abs/1412.3555

The paper compares gated recurrent units, including LSTM and GRU, on sequence tasks. It reports that gated units outperform traditional tanh recurrent units in the evaluated tasks and that GRU was comparable to LSTM. GRU's recurrent dependency makes training sequential across time, while its recurrent state gives constant-size inference memory with respect to sequence length.

## FlashAttention
Source: https://arxiv.org/abs/2205.14135

The paper distinguishes algorithmic quadratic self-attention from memory-traffic inefficiency. FlashAttention remains exact attention but uses tiling and IO awareness to reduce high-bandwidth-memory reads and writes. Reported speedups are hardware- and implementation-specific; this supports prioritizing fused/blocked kernels even when changing the model algorithm is not desired.

## Transformers are SSMs: Structured State Space Duality
Source: https://arxiv.org/abs/2405.21060

The authors develop a framework connecting SSMs and attention variants through structured semiseparable matrices and report Mamba-2-style cores that are 2–8x faster in their evaluated settings while remaining competitive with Transformers. This supports considering a recurrent/state-space branch or hybrid design rather than treating linear attention and SSMs as unrelated alternatives.

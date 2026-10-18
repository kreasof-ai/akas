# Mixture of experts

## Purpose

Route tokens to experts, construct route groups/tiles, project expert inputs and combine
weighted results with the shared branch. Conceptually: `moe(x, weights, routing) ->
out`.

## Backend choice

**Recommendation: Share routing/combination candidates and semantic contracts; retain
accelerated expert-projection specializations.**

- **Reason to share:** Grouping, top-k tie rules, route bookkeeping and weighted
  combination should have one definition. Much of the current routing/combine code uses
  indexing, reductions and arithmetic. Atomic route construction still needs backend
  compilation and ordering checks.
- **Reason to split:** Expert projection inherits linear's CUDA matrix/FP8 and Hopper
  instruction differences. Compact, grouped, persistent and split plans also have
  different intermediate layouts and resource requirements. Keep those optimized choices
  behind the same routed-projection feature instead of duplicating the entire MoE block.

The six-file structure is implemented. Consolidation preserves the existing
registered support; shared bindings alone do not establish new device support. See the [shared authoring and backend
guide](../../../../docs/operations.md)
for the fixed file roles and the distinction between construction,
compilation, execution and performance.

## Current code

| File | Responsibility |
| --- | --- |
| [__init__.py](__init__.py) | Package boundary. |
| [definition.py](definition.py) | Public composition, semantic/output contracts, execution policies, registrations and eligibility. |
| [kernel.py](kernel.py) | Shared TileLang computations and inline helpers. |
| [cuda.py](cuda.py) | CUDA specializations, shared bindings and approved CUDA registration inventory. |
| [webgpu.py](webgpu.py) | WebGPU specializations, shared bindings and approved WebGPU registration inventory. |
| [validation.py](validation.py) | Independent host expectations for qualification. |

`cuda.py` contains the current router, route grouping, expert algorithms, merges, combination and Hopper variant. Shared portable counterparts still need qualification/implementation; the common `kernel.py` is their designated home. `definition.lower` contains routed execution policies.

`encoded_moe` composes routing, packed expert projections, the shared expert and
combination for encoded checkpoints. Its CUDA `quantized_experts` callable uses
GEMV for one token and grouped tensor-core GEMM for larger batches/prefill.
Decode takes expert IDs; prefill takes compact counts/routes. These are different
input layouts for the same logical result. Encoded coefficients decode inside
the consuming tile; the model is not expanded to BF16 or requantized to FP8.
`block_m` is a qualified beam-search axis.

`quantized_routes` is shared integer bookkeeping with CUDA execution and WebGPU
shader coverage. Qwen's packed-expert graph remains CUDA-only because it uses
native BF16 activations. This distinction is explicit in the
registrations and [GGUF tests](../../../../tests/test_qwen_gguf.py).

`routed_moe` composes sigmoid routing with selection biases and no shared expert.
Its FP32 activation contract runs on CUDA and WebGPU; `biased_router` and
`routed_combine` share the same callable bodies. CUDA reuses the packed GEMV and
grouped GEMM projections. WebGPU uses `portable_experts`, which fuses coefficient
decoding into bounded reduction tiles. These are registered implementations of
the same `quantized_experts` operation. LFM2 MoE uses this composition.

Expert qualification keeps the synthetic bank packed and decodes at most 128
output rows at a time for the independent host expectation. Immutable inputs
share their original arrays. Validation memory therefore includes the packed
input bank and small working tiles, without a full floating expert-bank copy.

## How to extend

Define route indices, counts, padding, tie handling and output accumulation
independently of the matrix algorithm. Share suitable router/combine bodies, then port
codecs and expert projections. Add a usable end-to-end WebGPU plan before claiming MoE
parity; current CUDA-only coverage is a missing implementation, not a hardware rule.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Test repeated experts, empty buckets, tied scores, inactive tokens, padded routes,
compact versus grouped and split versus unsplit projections. See [MoE program
tests](../../../../tests/test_moe_speculative_program.py), [native pipeline
tests](../../../../tests/test_native_pipeline_policy.py) and [Hopper
tests](../../../../tests/test_qwen_hopper.py). Benchmark routing plus expert work plus
combination.

Return to the [operation index](../README.md).

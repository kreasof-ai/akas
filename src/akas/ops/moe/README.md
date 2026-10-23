# Mixture of experts

Single-token F32 expert projections use the codec-owned `dot_float` capability
when available. Both providers retain F32 activations, coefficients and
accumulators; the kernel owns routing, launch geometry, warp reduction and
inactive outputs. Packed block fields remain in quantization. Formats without
that capability keep the general reader path.

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
grouped GEMM projections, with a warp-local decode reduction candidate. CUDA
prefill search varies route rows, output columns, reduction depth and thread
count per tile. `compact_experts` requests the weight codec's `read_tile` to stage
FP16 coefficients; format-specific packed loads stay in quantization.
For prefill, the shared `expert_tile_map(counts)` returns expert IDs and local
tile indices for each supported row tile. Empty experts produce no active work;
one map serves gate, up and down projections on both providers. The map is
independent of the selected geometry, so search can reuse its validation inputs.
CUDA's `compact_experts` keeps tensor-core matrix tiles. WebGPU's
`register_experts` shares decoded coefficient tiles and keeps accumulators in
private registers. Its search axes include row/column/depth tiles, per-thread
microtiles, vector dot width and shared-memory layout/padding. `warp_experts`
uses subgroup reductions for decode. `portable_experts` remains a fallback with
bounded reduction tiles. These are registered implementations of the same
`quantized_experts` operation. LFM2 MoE uses this composition.
CUDA's `staged_experts` is another implementation of that operation. It asks the
weight codec for `stage_tile`, prefetches the compressed tile into shared memory,
then decodes it for the same FP16 tensor-core projection and F32 accumulation.
The codec defines byte layout; the operation defines tile sizes, thread count,
and pipeline stages. Eligibility follows codec capability and complete tile
geometry, with no model-name checks. The direct reader remains available for
other layouts and shapes. Larger shared-memory footprints can reduce occupancy,
so select staging and its geometry through qualification and measurement.
WebGPU tile selection respects the portable compiler's 32 KiB workgroup budget;
oversized combinations are rejected before lowering. The older `tiled_experts`
and `portable_experts` implementations retain their original counts/routes ABI.

Expert qualification keeps the synthetic bank packed and decodes at most 128
output rows at a time for the independent host expectation. Immutable inputs
share their original arrays. Validation memory therefore includes the packed
input bank and small working tiles, without a full floating expert-bank copy.

CUDA packed expert decode exposes columns per warp, thread count and depth-loop
unrolling. Activation values can be reused across output columns while retaining
FP16 operand rounding, FP32 accumulation and the same per-column summation
order. Larger tiles are candidates rather than presumed improvements; the
single-column schedule remains the default.

For eligible encoded weights, CUDA's `packed_experts` asks the weight codec for
`dot_half`: each warp consumes 256 input channels per iteration, rounds operands
to FP16 and accumulates in F32. Block decoding and packed loads belong to
`quantization/weights`, rather than this operation. A single unpermuted piece
with that capability can use the packed path; other layouts keep the general
expert implementation. This avoids allocating a dequantized expert bank or a
second weight cache. The codec receives the previous lane accumulator on every
tile, preserving continuous F32 summation. Adding independently rounded tile
dots can change a later router decision even when early model diagnostics pass.
Thread count remains a measured schedule choice.

The `fused_swiglu` policy can replace a private, adjacent single-token gate/up/
SwiGLU sequence with `expert_swiglu(x, gate, up, ids, counts) -> activation`.
The down projection stays separate, so output tiles do not recompute gate/up.
CUDA carries each codec's FP16 dot through continuous FP32 lane accumulators;
the shared implementation also supplies the WebGPU callable. Each provider
retains the original graph's operand-rounding policy. Projection values exposed
as features, views or additional consumers keep their original calls. The
independent reference samples weights through their codec and decodes bounded
row tiles. Measure the complete sequence and whole-model execution before
selecting a fusion schedule; cache-resident microbenchmarks are insufficient.

CUDA's `parallel_routes` compacts rows with warp prefix sums and one shared
cursor per expert. It preserves token/rank order and initializes unused entries
to -1, including fully inactive batches. The shared implementation retains the
same route-table contract for other providers.

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

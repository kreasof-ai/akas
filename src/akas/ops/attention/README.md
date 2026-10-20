# Attention

## Purpose

Causal attention over a KV cache, with preparation, cache updates, decode/prefill
algorithms and optional output gating. Conceptually: `attention(query, cache, sequence)
-> out`; preparation also returns updated cache state.

## Backend choice

**Recommendation: Share the operation contract and portable computations; retain
specialized accelerated algorithms.**

- **Reason to share:** Causal masking, head mapping, online-softmax mathematics, cache
  access contracts and partial/merge semantics should have one definition. Portable
  paths live together in `kernel.py`; a split/merge plan is still one
  public attention feature.
- **Reason to split:** CUDA decode uses warp shuffles and matrix schedules; WebGPU paths
  use subgroup externs, score/value partitioning and their own launch geometry. Native
  and Hopper paths have additional cache and tiling constraints. Keep these algorithms
  where useful instead of replacing them with one serial fallback. Bind the cache codec
  in quantization rather than branching on encoding throughout attention.

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

Shared preparation/cache/online/floating-decode functions are in `kernel.py`. CUDA decode, prefill, partial/merge and Hopper variants are together in `cuda.py`; subgroup score/value algorithms are in `webgpu.py`. The split/decoded-cache policy is `definition.lower`.

CUDA floating prefill offers a tensor-core load pipeline for native FP16 KV.
The model declares `unit_scales` only when its cache writer stores unit metadata;
generic scaled caches keep their scaling computation. Query/key tile sizes,
load pipelining, stage count and thread count are explicit schedule choices.
Constant groups of four key tiles keep asynchronous waits static even when
sequence lengths are dynamic. The outer loop advances those groups; online
softmax remains ordered. Merely changing a serial loop annotation does not make
loads asynchronous, and additional stages can reduce occupancy. Qualify and
measure the generated code before promoting a schedule.

Floating decode exposes its KV partition count through the attention graph
policy. The policy updates the partial buffer and both producer and merger
contracts together. Grouped decode shares each KV tile across query heads and
updates online softmax per tile. Its value depends on the provider and shape;
compatible implementations remain available for measured selection.

CUDA's native-half `mma_partial` shares a KV tile across a group of query heads
and uses tensor-core QK/PV products. Queries and probabilities round to FP16;
softmax statistics and accumulated values stay F32. The `unit_scales` contract
selects this path only when native cache scales are one. Other cache contracts
retain their compatible implementations.

Complete groups of four key tiles use `T.Pipelined`. At most four final tiles
use guarded loads, keeping unused or nonfinite future cache entries out of PV.
Separating the tail also avoids asking the compiler to pipeline a dynamically
sized final group. Partial buffers and the merge ABI stay the same; the
independent oracle accounts for tile-aligned partitions and FP16 probability
rounding. These implementations declare `key_rows` in `reference_keys`, so
tuning cannot reuse an oracle with different tile rounding or split boundaries.
See [decode qualification](../../../../tests/test_attention_mma_decode.py).

WebGPU's `tiled_prefill` shares key/value tiles across query rows and grouped
heads. It performs online softmax per key tile and retains the output
accumulators in private registers. The score and value loops load a KV vector
once and reuse it across query rows. Query rows, key rows, staging precision,
padding and dot width are qualified search axes. Native FP16 KV permits half
staging without changing stored values; other cache formats use FP32 staging.
The launch stays within 32 KiB of workgroup storage. Decode and ineligible
prefill shapes continue through their registered attention implementations.

Long-context RoPE declares `position_limit`. Beyond 8K, it uses split inverse
frequencies and compensated angle reduction before sine/cosine. CUDA uses fused
multiply-add for product errors; WebGPU splits operands because fused rounding
is not guaranteed there. Exact bit-pattern constants avoid loss through WGSL
decimal formatting. Both implement the same rotation and qualify against a
double-precision host reference, including the final context position.

## How to extend

State query/cache shapes, mask semantics, cache mutations and output gating before
choosing an algorithm. For a new split algorithm, declare the partial values and
normalization-statistics layout and provide its merge. Add an equivalent feature to both
providers without requiring identical internal launches. Existing native preparation,
mapped/cache ABIs and pipeline coverage are uneven; shared helpers do not establish full
parity.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Test decode and prefill, context tails, masked/future entries, inactive requests, cache
scales, split versus unsplit results and gating. Start with [attention pipeline
tests](../../../../tests/test_attention_pipeline.py), [KV
tests](../../../../tests/test_kv_quantization.py) and [WebGPU lowering
tests](../../../../tests/test_webgpu_model_lowering.py). Benchmark the complete
preparation/attention/merge plan.

Host prefill qualification computes online softmax in bounded key tiles. Batched
matrix products reuse query groups without constructing a full query-by-context
score matrix. The reference retains each implementation's declared operand and
probability precision, including FP16 rounding between tensor-core tiles.
Large prefill references use F32 working arrays; small cases and decode retain
F64. Cross-checks compare both host paths with identical fixtures and unchanged
tolerances. Single-block cache scales broadcast over channels, avoiding a full
materialized scale matrix. GPU arithmetic is independent of this host policy.

For mapped native attention, only live rows of a freshly returned output are
defined. Padding rows are not borrowed state and need not preserve old buffer
contents. Qualification compares live output rows numerically and checks all
inputs and KV state in full.

Return to the [operation index](../README.md).

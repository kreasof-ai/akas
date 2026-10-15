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

For mapped native attention, only live rows of a freshly returned output are
defined. Padding rows are not borrowed state and need not preserve old buffer
contents. Qualification compares live output rows numerically and checks all
inputs and KV state in full.

Return to the [operation index](../README.md).

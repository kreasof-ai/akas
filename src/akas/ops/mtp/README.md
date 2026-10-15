# MTP tensor helpers

## Purpose

Cast hidden values and join separately normalized embedding/hidden tensors for the
existing Qwen MTP path. Conceptually: `mtp_cast(x) -> out` and `mtp_join(embedding,
hidden, weights) -> joined`.

## Backend choice

**Recommendation: Share these tensor computations; keep Qwen-specific model wiring
outside generic operation infrastructure.**

- **Reason to share:** The current bodies contain casts, parallel arithmetic and
  `T.reduce_sum`, with no explicit CUDA intrinsic. The normalized join includes
  zero-centered weights and BF16 output behavior, which should be defined once.
- **Reason to split:** CUDA placement and the MTP model name are not reasons to copy a
  cast or normalization. Specialize a reduction or schedule only if target qualification
  or measurements justify it.

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

`kernel.py` contains `mtp_cast` and `mtp_join`; both backend files expose the same shared helpers. Existing BF16/block-FP8 registrations remain CUDA-only. The model-specific definition stays in [Qwen MTP](../../models/qwen35/mtp.py).

## How to extend

Keep the helper's tensor contract separate from a speculative-head framework. A new
reusable cast/join should expose its own shapes, centering and precision semantics.
Qualify existing bodies on WebGPU before adding equivalent registrations; this
documentation does not add that support.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check each normalization separately, zero-centered weights, concatenation order, epsilon
and BF16 rounding. See [Qwen MTP tests](../../../../tests/test_qwen_mtp.py).

Return to the [operation index](../README.md).

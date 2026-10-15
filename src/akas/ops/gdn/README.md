# Gated delta network

## Purpose

Prepare and apply a gated delta recurrence with convolution history and recurrent matrix
state. Conceptually: `gdn(operands, state, sequence) -> (state, out)`; verification can
return tentative results/checkpoints without committing persistent state.

## Backend choice

**Recommendation: Share core preparation/recurrence computations first; keep state plans
and any justified accelerated schedules explicit.**

- **Reason to share:** The current convolution, preparation, normalization and scan
  bodies use TileLang loops, copies, arithmetic and reductions rather than explicit
  warp/WGMMA intrinsics. Sharing their source avoids duplicating the recurrence. Shared arithmetic makes state-update and rounding fixes apply to both
  providers.
- **Reason to split:** A different parallel scan, resource limit, conversion lowering or
  measured schedule can justify a specialized implementation after qualification.
  Preserve existing recurrent/scan algorithms and launch geometry while assessing
  sharing; do not replace them with a naive serial implementation. Snapshot,
  recomputation and deferred-journal policies have distinct state contracts even when
  shared across providers.

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

Convolution, preparation, scan, normalization, journals and restoration now live together in `kernel.py`. Both backend modules bind these functions; only existing CUDA cases are registered. `definition.lower` owns the snapshot/recompute policy.

## How to extend

Separate the mathematical recurrence from request mapping, logical dimensions and commit
policy. Introduce a shared candidate with the existing arithmetic before changing
schedules. Compile and execute BF16/reduction cases on WebGPU before registering
support. Port preparation, scan and commit/recovery together so a model has a usable
feature rather than an isolated recurrent kernel.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Compare token-by-token recurrence with chunked execution and test nonadjacent request
mappings, inactive rows, unselected pool entries, and zero/partial/full acceptance.
Start with [Qwen kernel tests](../../../../tests/test_qwen_kernels.py), [recompute
tests](../../../../tests/test_qwen_recompute.py) and [state
tests](../../../../tests/test_state_prefill.py). Measure the full state plan, including
recovery.

Return to the [operation index](../README.md).

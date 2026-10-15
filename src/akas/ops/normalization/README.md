# Normalization

## Purpose

RMS normalization, residual-add/RMS fusion and Q/K normalization variants. Conceptually:
`rms(x, weight) -> out` or `add_rms(x, residual, weight) -> (residual, out)` when the
residual is mutated.

## Backend choice

**Recommendation: Share normalization mathematics and bind the appropriate reduction
primitive/schedule.**

- **Reason to share:** Centering, epsilon placement, residual writes and rounding are
  common semantics. `kernel.py` already contains fragment, serial-tree and parallel-tree
  variants. Different reduction orders do not require separate copies of all
  normalization logic.
- **Reason to split:** WebGPU subgroup externs justify a specialized reduction helper. A
  target-dependent fragment lowering or measured launch choice may justify a schedule.
  Keep those differences local, and preserve the existing precision contract instead of
  assuming every RMS variant is interchangeable.

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

Fragment, parallel-tree and serial-tree algorithms remain in `kernel.py`; target subgroup helpers live in [kernel infrastructure](../../kernels/README.md). Fused graph rewrites are in `definition.py`.

## How to extend

State whether weights are zero-centered, when inputs/residuals round and which buffer is
updated. Bind a backend reduction helper instead of importing WebGPU intrinsics into an
otherwise shared path. Add qualification recipes for new variants; `qnorm`/`kvnorm`
currently lack local registry recipes.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Test ordinary and zero-centered weights, large/small values, residual alias/write
behavior, tails and intentional rounding. Use [normalization parity
tests](../../../../tests/test_normalization_parity.py) and [operation write
tests](../../../../tests/test_operation_writes.py).

Return to the [operation index](../README.md).

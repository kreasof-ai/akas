# Residual addition

## Purpose

Add two float32 buffers with `r * c` elements. Conceptually: `add(x, y) -> out`.

## Backend choice

**Recommendation: Keep one shared TileLang computation.**

- **Reason to share:** The current body uses parallel indexing, a bounds mask and `x +
  y`. Both providers already select it. One source keeps tail handling and arithmetic
  consistent with no extra dispatch or launch.
- **Reason to split:** There is no source-level need for separate CUDA/WebGPU bodies. A
  different dtype, alias contract or useful fusion must be declared explicitly; a
  filename change is not an optimization.

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

`kernel.add` is one shared float32 addition body. Both backend files bind it directly.

## How to extend

Use this family as the smallest authoring example. Keep the flattened tensor contract
explicit. For a new fused operation, declare every observable result/state write and
preserve callers that still need the unfused residual.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Test widths smaller than a block and nonmultiples of 256, multiple rows and unchanged
inputs. The [new-operation
walkthrough](../../../../docs/operations.md#add-a-shared-operation)
builds on this format; [operation write
tests](../../../../tests/test_operation_writes.py) cover graph-visible effects.

Return to the [operation index](../README.md).

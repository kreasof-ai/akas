# Activation

## Purpose

Elementwise SwiGLU for separate gate/up tensors, packed projections, and routed expert
rows. Conceptually: `swiglu(gate, up) -> out`.
`sigmoid_bias(x, bias) -> out` applies a shared scalar bias and sigmoid. Its shared
CUDA/WebGPU body is used by draft confidence heads.

## Backend choice

**Recommendation: Use one shared computation, with explicit layout and rounding variants.**

- **Reason to share:** The existing `_swiglu` body uses elementwise arithmetic, indexing
  and casts. Packed columns and expert rows change the tensor contract, not the hardware
  algorithm. Sharing keeps the sigmoid formula and BF16 rounding rules consistent.
- **Reason to split:** A separate whole CUDA/WebGPU implementation has no demonstrated
  justification here. If a measured vectorization improvement needs a specialized
  schedule, keep the element formula shared and declare the schedule's eligible shapes.

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

`swiglu` and `swiglu_experts` share `_swiglu` in `kernel.py`; provider defaults remain in `definition.py`.

## How to extend

Add a named tensor layout or numerical variant to the shared computation. Keep expert
routing separate from activation arithmetic. Register equivalent supported cases for
both providers; current expert-row and native BF16/block-FP8 routes are CUDA-only.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check packed versus separate inputs, expert rows, nonmultiple widths and explicit
rounding. Use [activation parity tests](../../../../tests/test_activation_parity.py) and
the independent recipe; adding a registration does not prove a new dtype compiles.

Return to the [operation index](../README.md).

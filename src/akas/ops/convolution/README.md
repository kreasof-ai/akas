# Convolution

## Purpose

LFM-style gated short convolution with persistent history. Conceptually: `convolution(x,
weights, history, enabled) -> (history, out)`. Its gating/history contract differs from
GDN preparation convolution.

## Backend choice

**Recommendation: Keep the current shared computations.**

- **Reason to share:** Both providers already select `convolution` or `convolution_flat`
  from `kernel.py`. They use parallel channel work and serial token/history updates with
  no explicit backend intrinsic. Flat decode and batched chunks are layout/state
  contracts.
- **Reason to split:** Separate backend bodies would duplicate the history update and
  gate arithmetic. Specialize only if a new measured convolution algorithm warrants it,
  while retaining the same history order and inactive-token behavior.

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

Both reference/chunk and flat-history bodies are in `kernel.py`. The empty legacy CUDA package has been removed.

## How to extend

Start with the shared body and state contract. A different tap count or gate layout
needs explicit shapes and independent expectations. Do not treat GDN's convolution as an
interchangeable variant merely because both use history buffers.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check chunked execution against repeated token execution, history ordering, short/empty
enabled prefixes and channel tails. See [state prefill
tests](../../../../tests/test_state_prefill.py) and [LFM reference
tests](../../../../tests/test_lfm2_reference.py).

Return to the [operation index](../README.md).

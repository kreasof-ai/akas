# Sequence and row control

## Purpose

Build positions/masks, gather or store hidden state, select last rows/tails and advance
sequences. Each callable declares the control or persistent-state buffers it changes.

## Backend choice

**Recommendation: Share the indexing/state computations, with explicit ABI variants.**

- **Reason to share:** The shared `kernel.py` already serves both providers. The flat and mapped variants now live together in `kernel.py`. Their indexing, integer arithmetic and masks do not require
  backend-owned copies.
- **Reason to split:** A genuinely faster gather/vectorized copy can warrant
  specialization after measurement. Mapped versus flat requests and different
  position-update rules are contracts, not backend identities; reconcile those before
  sharing superficially similar bodies.

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

Row movement, positions, control masks, native/flat ABI variants and both flat-tail schedules are now together in `kernel.py`. Backend files expose shared bindings; `definition.py` shows which cases are approved for each provider.

## How to extend

Name the logical operation rather than creating another backend-specific `rows`
convention. Specify whether positions were already advanced by argmax, how inactive rows
behave, and how mappings address saved state. Keep thread-binding differences as
schedules when possible.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check zero/partial/full lengths, last-row off-by-one rules, tails, inactive masks,
nonadjacent requests and untouched saved rows. See [sequence tail
tests](../../../../tests/test_sequence_tail.py), [causal tail
tests](../../../../tests/test_causal_tail.py) and [device feedback
tests](../../../../tests/test_device_feedback.py).

Return to the [operation index](../README.md).

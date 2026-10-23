# Indexing

## Purpose

Clear selected persistent-state rows through a logical-row-to-request mapping.
Conceptually: `clear_rows(mapping, state) -> state`.
`select_token(x) -> out` gathers one fixed token position from each row;
`replace_token(tokens, replacement) -> out` creates a block with one column
replaced; `concatenate(left, right) -> out` joins feature columns in each row.
These functions leave their input matrices unchanged.

## Backend choice

**Recommendation: Keep one shared computation.**

- **Reason to share:** `clear_rows` already serves both providers. Its address
  calculation, mapping checks, tail mask and zero stores are portable computation rather
  than a hardware-specific algorithm.
- **Reason to split:** Backend copies would add maintenance without a current
  instruction-level reason. A measured vectorized-store improvement can specialize
  scheduling while sharing the mapping and bounds contract.

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

All four bodies are shared in `kernel.py`; backend modules reference them directly.

## How to extend

Add another indexing operation as a typed callable with explicit state results. Specify
handling of invalid mappings and index bounds. Keep request mapping separate from
checkpoint/model names.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check negative/out-of-range mappings, nonadjacent requests, channel tails and unchanged
unselected rows. See [state mapping tests](../../../../tests/test_state_mapping.py) and
[shared execution tests](../../../../tests/test_shared_execution.py). The
[draft primitive tests](../../../../tests/test_draft_primitives.py) check immutable
gathers, column replacement and concatenation on CUDA and WebGPU.

Return to the [operation index](../README.md).

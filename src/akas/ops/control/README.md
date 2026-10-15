# Control

## Purpose

Buffer copy, token feedback and speculative-round bookkeeping. Inputs and results
include control tensors, tokens, lengths and acceptance results; these operations do not
define a speculative model head.

## Backend choice

**Recommendation: Use shared TileLang computations; preserve explicit bookkeeping contracts.**

- **Reason to share:** Copy and feedback already share implementations. The six
  functions in `kernel.py` use indexing, comparisons and scalar control rather than
  explicit CUDA intrinsics. Their bookkeeping needs no backend-specific arithmetic.
- **Reason to split:** Do not duplicate the bookkeeping for each provider without a
  concrete primitive or performance need. Any future device-specific scheduling should
  retain the same accepted-prefix and control-buffer behavior.

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

Copy, feedback and all six speculative-round bodies are together in `kernel.py`. `definition.py` still registers the speculative kinds for CUDA only; a shared binding is not a claim of qualified WebGPU support.

## How to extend

Document every control field and whether a result overwrites or preserves previous
contents. Keep acceptance bookkeeping separate from proposer/head definitions. Share a
body first and qualify the same cases on both providers; the current speculative routes
are only registered for CUDA.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Use exact comparisons for integers and verify unchanged inputs. Cover zero proposals,
mismatches at each position, full acceptance and inactive rows. See [device feedback
tests](../../../../tests/test_device_feedback.py) and [speculative decoding
tests](../../../../tests/test_speculative_decoding.py).

Return to the [operation index](../README.md).

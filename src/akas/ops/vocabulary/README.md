# Vocabulary

## Purpose

Embedding lookup, deterministic token argmax and accelerated vocabulary-head
projection/reduction. Native argmax variants can also update positions; that effect must
remain explicit.

## Backend choice

**Recommendation: Share lookup and argmax computation where their contracts match;
retain accelerated fused-head specializations.**

- **Reason to share:** Lookup can consume a format-owned storage/decoder binding. Parent
  and WebGPU packed embedding bodies already have very similar parallel lookup loops.
  WebGPU argmax also delegates to a parent implementation with an unrolling attribute.
  Those differences do not justify independent whole-operation trees.
- **Reason to split:** CUDA's tiled head projects and reduces without materializing all
  logits, so its optimized matrix/merge plan should remain. Current native BF16 argmax
  has active masking and position increments unlike some flat/reference variants;
  matching semantics is required before sharing. Byte versus word-packed storage belongs
  to quantization, not embedding's mathematical definition.

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

`kernel.py` contains shared/reference, native-masked and packed-word lookup/argmax variants. Packed lookup now consumes a [quantization-owned weight reader](../../quantization/weights/__init__.py). `cuda.py` contains the optimized head projection/reduction stages; the corresponding WebGPU head remains a feature gap.

`embed(...)` accepts a native buffer or encoded `Weight` descriptor through one
composition entry. `quantized_embedding` decodes only requested rows through the
format reader, returning BF16 for the native Qwen graph or F32 for portable
consumers. It does not allocate a decoded vocabulary matrix.


For large materialized F32 logits, the `vocabulary.parallel_argmax` compiler
policy splits plain `argmax` into `argmax_partial` and `argmax_merge`. Both
providers use the same TileLang factories. The merge preserves the first index
on ties and leaves inactive outputs untouched; all-invalid rows retain the
legacy vocabulary-size sentinel. This policy does not rewrite fused heads or
native argmax variants that update positions. Scratch buffers and the two
stages belong to the compiled program, so serving uses the same selected plan.

## How to extend

Define `embedding`, token selection and head projection as clear callable contracts. Put
new formats, scales and packed readers in quantization and consume a resolved codec.
Preserve inactive-token behavior, deterministic ties and position side effects when
comparing variants. The fused head currently has CUDA coverage only; a matching WebGPU
feature needs an implementation and qualification.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check encoding types, token bounds, partial vocabulary ranges, equal-score ties,
inactive requests, position updates and fused versus materialized head outputs. Use
[GGML validation tests](../../../../tests/test_ggml_validation.py), [Qwen kernel
tests](../../../../tests/test_qwen_kernels.py) and [device feedback
tests](../../../../tests/test_device_feedback.py).

Return to the [operation index](../README.md).

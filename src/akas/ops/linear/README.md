# Linear projection

## Purpose

Dense and paired feed-forward projections, optional residual fusion, quantized operands
and split-reduction plans. Conceptually: `linear(x, weights) -> out`; paired/fused
variants have additional explicit results or effects.

## Backend choice

**Recommendation: Share contracts and portable arithmetic; retain accelerated backend algorithms.**

- **Reason to share:** The portable projection and common epilogue mathematics can be
  shared. Selection by rows, dimensions, weight representation and output view should be
  explicit rather than hidden behind model identity. Shared helpers can also serve dense
  and routed projections.
- **Reason to split:** CUDA uses warp reductions, packed hardware loads, MMA and Hopper
  WGMMA. WebGPU uses subgroup reductions, vector externs and packed-integer matmul.
  These are concrete instruction/layout/schedule differences. The `cuda.py:_mma`
  helper is shared between consumers but still contains CUDA-oriented FP8/external
  operations; its name does not prove cross-backend portability.

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

Portable projections and epilogue/ABI helpers are in `kernel.py`. CUDA GEMV/GEMM, shared dense/expert FP8 MMA, split/merge and Hopper variants are in `cuda.py`. WebGPU subgroup/integer implementations are in `webgpu.py`. `definition.py` contains projection composition, scheduling, rewrites and split policy.


Small F32 vector projections can use `column_projection(spec, schedule)` on
both providers. One workgroup reduces each output column; `threads` controls the
F32 reduction geometry. WebGPU selects it for single-row, unencoded matrices
with at most 256 outputs and depth at least 128. CUDA keeps its measured warp
implementation as the default; the shared alternative remains selectable.
Changing the reduction geometry needs numerical and model qualification.

## How to extend

Declare the operand storage, accumulation/output precision and any residual effects
before adding an algorithm. Put new encoding arithmetic and layout rules in
quantization. Implement conversion and consuming projection together for a new format.
Preserve the existing eligible GEMV/GEMM selection; shared source must not force every
shape through the same slow algorithm.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check small-row GEMV and larger GEMM, tails, paired outputs, fused residuals, block
scales, output views and split merges. Use [packed fusion
tests](../../../../tests/test_packed_fusions.py), [Q16
tests](../../../../tests/test_activation_q16.py), [kernel
tests](../../../../tests/test_qwen_kernels.py) and [WebGPU
tests](../../../../tests/test_webgpu_model_lowering.py). Benchmark complete
conversion/projection/merge plans.

Return to the [operation index](../README.md).


## Encoded operands through the common codec interface

`quantized_linear` is an additive operation under the `quantized` contract.
It accepts F32 or FP8/Q8/Q16 activation storage and GGML or block-FP8 weights on
CUDA and WebGPU. Its kernel resolves `Format.bind` readers, uses GEMV for one
row and reusable decoded FP16 tiles for GEMM, and accumulates in F32. Payloads
and required scale buffers are explicit inputs; Q8/Q16 correction sums are only
needed by packed integer consumers. No full decoded operand is allocated.

CUDA's existing packed prefill path also requests weight `read_tile` and
`read_pair` through this binding. Linear, paired FFN and residual projections
therefore share format-specific load optimizations while retaining their
tensor-core schedules and fused epilogues. Codec word views preserve each
input's pointer identity, including the two separate FFN weight matrices.

Existing tuned packed/integer/native FP8 registrations remain separate and keep
their original algorithms. This common consumer supplies compatibility; it is
not a claim that every shape outperforms a specialized native kernel. FP16
staging has its usual rounding/range and requires shader FP16 on WebGPU.

See the [quantization guide](../../quantization/README.md) for a complete graph
example and [measurement guide](../../../../docs/benchmarking.md).

`quantized_projection` consumes `quantization.weights.Weight`, which describes
one logical matrix with up to four physical pieces. Each piece has its own format
reader and optional group permutation. `project(...)` accepts this descriptor or
a native buffer through the same entry. Combined QKV/GDN/FFN projections therefore
do not need to unpack or requantize mixed checkpoint encodings at load time.

The computation shares GEMV and tiled GEMM with the common codec consumer. CUDA
prefill tunes its M/N/K tile sizes and uses FP16 tensor-core operands with F32
accumulation. WebGPU has F32-input shader coverage; Qwen's BF16 activation graph
still requires CUDA. See [GGUF qualification tests](../../../../tests/test_qwen_gguf.py).

Native F32 projections have a CUDA warp specialization, useful for small
routers as well as larger row batches. It assigns one output column to each
warp and replaces shared reduction barriers with warp shuffles. Both operands
and accumulation stay F32; this path does not quantize router weights. Output
strides, offsets and partial columns retain the common projection contract.


# Operations and kernels

Every operation family now has the same six Python files and this README:

```text
ops/<family>/
    __init__.py
    definition.py
    kernel.py
    cuda.py
    webgpu.py
    validation.py
    README.md
```

| File | Responsibility |
| --- | --- |
| `__init__.py` | Package boundary. |
| `definition.py` | Public graph composition, contracts/effects, execution policies, implementation registrations and tuning eligibility. |
| `kernel.py` | Shared TileLang computations and inline helpers. |
| `cuda.py` | CUDA specializations, shared bindings and approved CUDA choices. |
| `webgpu.py` | WebGPU specializations, shared bindings and approved WebGPU choices. |
| `validation.py` | Independent numerical/state expectations, including host references. |

Start with the [beginner authoring guide](../../../docs/operations.md).
Its complete example uses the actual six-file structure and current discovery API.

## Share a computation or specialize it

Prefer one TileLang body used by both providers. A backend file can bind the same
shared function; matching structure does not require copying its arithmetic.
Specialize only when target instructions, physical layouts, launch plans or
measured performance justify a different implementation. Both versions retain
one public semantic contract, with explicit masks, precision and state effects.

| Family | Decision and extension instructions |
| --- | --- |
| [activation](activation/README.md) | Shared SwiGLU. |
| [attention](attention/README.md) | Shared semantics/portable bodies; accelerated backend algorithms. |
| [control](control/README.md) | Shared copy, feedback and speculative bookkeeping. |
| [convolution](convolution/README.md) | Shared short convolution. |
| [gdn](gdn/README.md) | Shared preparation, recurrence and state helpers; explicit execution policy. |
| [indexing](indexing/README.md) | Shared mapped row operations. |
| [linear](linear/README.md) | Shared portable projection; specialized accelerated GEMV/GEMM. |
| [moe](moe/README.md) | Shared routing/combine candidates; specialized expert projection. |
| [mtp](mtp/README.md) | Shared cast and normalized join; Qwen-specific model wiring stays in models. |
| [normalization](normalization/README.md) | Shared RMS/fusion with reduction strategy selection. |
| [residual](residual/README.md) | Shared addition. |
| [sequence](sequence/README.md) | Shared row, position and control computations. |
| [vocabulary](vocabulary/README.md) | Shared lookup/selection; accelerated fused-head specialization. |

## Write a callable computation

Every standalone device definition uses typed tensor inputs and explicit results
through `@kernel`. Its `(spec, schedule)` factory binds geometry and tuning, then
returns the computation. A state input that is mutated is also returned. Fresh
results use `output(...)`; this is a declaration, not an execution-time allocation.
See the [functional contract](../../../docs/kernels.md).

A public operation may select a multi-launch algorithm. Split attention has
partial and merge callables with explicit intermediate layouts. GDN verification
has explicit tentative-state and commit behavior. Uniform functions do not force
these plans into one naive kernel.

`definition.py` owns `IMPLEMENTATIONS`, `VALIDATIONS`, output effects and policies.
The registry discovers these definitions directly. Stable tuning identifiers are
preserved even when factory paths move. Existing optimized algorithms remain
eligible under their original contracts, geometry and target restrictions.

## Infrastructure and quantization

Authoring, registry and selection infrastructure lives in
[kernels](../kernels/README.md), outside operation families. The public
`from akas.ops import KernelSpec, Capabilities, define` entry remains available.
Request-state mapping lives in
[compilation/transforms/state.py](../compilation/transforms/state.py).
Inline reductions live in [kernel primitives](../kernels/README.md#inline-primitives).

[Quantization](../quantization/README.md) uses the same six-file format package
structure. It owns encoded storage, packing, metadata and codec rounding. Packed
lookup and portable packed projection consume a quantization-owned weight reader;
GGUF parsing consumes quantization's encoding definitions.

## What is validated

The structure is implemented and enforced by
[structure tests](../../../tests/test_ops_structure.py). Each feature needs
explicit registration and device qualification on its supported providers. Inspect `definition.py`, backend `IMPLEMENTATIONS`
and codec `FORMAT.providers` for actual coverage. In particular, the current
WebGPU compiler profile does not directly accept native uint8/BF16 buffers used
by several CUDA paths; a counterpart needs compatible storage and conversion.
Each family README describes its backend choices and qualification requirements.

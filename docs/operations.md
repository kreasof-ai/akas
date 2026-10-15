# Writing operations for CUDA and WebGPU

Write one TileLang computation for both backends whenever its arithmetic, buffer
contract and device primitives can be shared. Keep a specialized implementation
when a concrete instruction, layout, launch plan or measured performance reason
requires it. A directory name is not evidence that code must be backend-specific.

The [quantization guide](../src/akas/quantization/README.md) describes storage and
codec interfaces. Each [operation README](../src/akas/ops/README.md) explains its
backend choices and actual supported contracts.

## Why the same language can still need different implementations

Both paths construct TileLang kernels, but this repository sends them through
different compiler backends: [artifact emission](../src/akas/compilation/emit.py)
selects NVRTC for CUDA and WGSL for WebGPU. The language used to describe a kernel
does not supply an equivalent for every external function or hardware instruction.

For example, CUDA [decode attention](../src/akas/ops/attention/cuda.py)
uses `T.shfl_down` and `T.shfl_sync`. WebGPU
[score computation](../src/akas/ops/attention/webgpu.py) uses explicit
`subgroupShuffleXor` extern calls. The CUDA
[Hopper projection](../src/akas/ops/linear/cuda.py) uses
`T.wgmma_gemm`, `T.wait_wgmma` and imported CUDA source. These are concrete
specialization boundaries. By contrast,
[residual addition](../src/akas/ops/residual/kernel.py) already serves both
providers through one body using indexing and parallel arithmetic.

Ordinary-looking constructs still need qualification. A shared `T.gemm`, reduction,
BF16 conversion or atomic operation is not proof that the current compiler can
lower every dtype, layout and launch geometry on both targets. Compile and run
the supported cases before declaring parity.

## The two choices

| Choice | Reasons to choose it | Cost and evidence required |
| --- | --- | --- |
| One shared computation | Same tensor/state contract; same algorithm; supported primitives. One arithmetic change updates both providers, and a beginner learns one implementation. | Verify lowering and execution on both providers. Allow different tuning values without copying the computation. A shared implementation must not replace an existing faster eligible route merely to remove a file. |
| Backend specializations of one operation | Different supported intrinsics, layout-sensitive vector access, launch decomposition, or a demonstrated performance advantage. The public operation and effects remain the same. | More code and duplicated maintenance. State the exact difference and validate both paths against the same independent expectations. Sharing the language, or having a historical `cuda/` directory, is not a reason to duplicate. |

Most complex families combine these choices: shared semantics and portable
computations, with specialized algorithms or small inline primitives. Avoid a
single giant computation with backend branches throughout its loops. Bind a
decoder/reduction/tile primitive during kernel construction when that is the only
difference; it stays inline and does not require another device launch.

## Decisions for every current directory

Shared arithmetic can have different backend registrations. Check each
registration and its device qualification before relying on a backend.

| Directory | Decision | Concrete reason |
| --- | --- | --- |
| [activation](../src/akas/ops/activation/README.md) | Share computation | SwiGLU differs by tensor layout and rounding options, not by a backend-specific algorithm. |
| [attention](../src/akas/ops/attention/README.md) | Share semantics/helpers; specialize accelerated algorithms | Warp versus subgroup reductions, native cache/request layouts, matrix tiling and split/merge plans differ materially. |
| [control](../src/akas/ops/control/README.md) | Share computation | Copy, feedback and speculative bookkeeping are ordinary indexing/control operations; no source-level CUDA intrinsic justifies the current whole-family split. |
| [convolution](../src/akas/ops/convolution/README.md) | Share computation | Both providers already use shared short-convolution bodies. State layout/chunk variants are semantic contracts. |
| [gdn](../src/akas/ops/gdn/README.md) | Share recurrence and preparation first | Existing core kernels use loops, copies, reductions and arithmetic. Keep state/verification plans explicit; preserve special schedules if qualification shows they need specialization. |
| [indexing](../src/akas/ops/indexing/README.md) | Share computation | Row clearing already uses the same body and registration shape for both providers. |
| [linear](../src/akas/ops/linear/README.md) | Share semantics/portable bodies; specialize accelerated algorithms | CUDA warp/MMA/WGMMA and WebGPU subgroup/integer-matmul paths have concrete primitive and scheduling differences. |
| [moe](../src/akas/ops/moe/README.md) | Share routing/combine candidates; specialize expert projections | Expert projection inherits linear's hardware differences; routing and combination do not intrinsically require separate mathematical implementations. |
| [mtp](../src/akas/ops/mtp/README.md) | Share helper computations | Cast and normalized join use conversions, arithmetic and reductions. Qwen-specific model wiring is separate. |
| [normalization](../src/akas/ops/normalization/README.md) | Share computation; bind reduction strategy | Preserve centering, residual writes and rounding. Backend subgroup primitives can specialize reduction without duplicating RMS semantics. |
| [reductions](../src/akas/kernels/README.md#inline-primitives) | Share tree helper; specialize device primitive | A shared-memory tree is reusable; CUDA warp and WebGPU subgroup intrinsics differ. These are inline building blocks. |
| [residual](../src/akas/ops/residual/README.md) | Share computation | Elementwise addition already has one body used by both providers. |
| [sequence](../src/akas/ops/sequence/README.md) | Share computation | Gather/store, masks, positions and tails are indexing/state operations; current native ABI differences do not establish a need for backend-owned copies. |
| [state](../src/akas/compilation/transforms/state.py) | One compiler transform | The function rewrites compiler IR; it is not a CUDA/WebGPU tensor kernel. |
| [vocabulary](../src/akas/ops/vocabulary/README.md) | Share lookup/argmax; specialize accelerated head | Lookup can consume a bound codec; masking and position writes need explicit variants. Fused head projection/reduction benefits from hardware-specific matrix schedules. |

## File structure

Every operation family and each of the six codec packages has exactly these
runtime files, plus its README:

```text
__init__.py        package boundary
definition.py     signatures, effects, composition, policies and registrations
kernel.py         shared portable computations
cuda.py           CUDA implementations and shared bindings
webgpu.py         WebGPU implementations and shared bindings
validation.py     independent numerical and state expectations
```

Both backend files can bind the same shared function. Their `IMPLEMENTATIONS`
inventories show approved cases. A shared binding is not a new claim of device
support. Keep registration, contracts, composition and lowering in these roles.

Authoring, selection and primitives live in [kernel infrastructure](../src/akas/kernels/README.md).
State mapping lives in [compilation transforms](../src/akas/compilation/transforms/state.py).
The registry discovers `definition.py` for operations and codecs; format lookup
also reads `definition.py`. A family is discovered through this boundary.

## Add a shared operation

Start from [residual](../src/akas/ops/residual/README.md) or
[indexing](../src/akas/ops/indexing/README.md). A new `square` family computes
`square(x) -> out` over `r * c` float32 elements. Create
`src/akas/ops/square/__init__.py` with a package docstring and the following files.
TileLang imports stay inside the factory so discovery needs no compiler.

In `kernel.py`:

```python
from akas.kernels.kernel import Tensor, kernel, output


def square(spec, schedule):
    import tilelang.language as T

    rows = spec.parameters.get("r", 1)
    count = rows * spec.parameters["c"]

    @kernel
    def compute(x: Tensor((count,), "float32")) -> Tensor((count,), "float32"):
        out = output((count,), "float32")
        with T.Kernel(T.ceildiv(count, 256), threads=256) as block:
            for lane in T.Parallel(256):
                index = block * 256 + lane
                if index < count:
                    out[index] = x[index] * x[index]
        return out

    return compute
```

Geometry changes the tensor contract; tuning must not overwrite it. `output()`
declares a result buffer rather than allocating one during a device launch.
A mutated state input must also appear in the explicit result tuple. See the
[functional contract](kernels.md) for aliases and output ABI ordering.

In `validation.py`, compute expectations without calling the device kernel:

```python
from akas.kernels.contracts import ValidationCase, sample_arguments


def validation(kind, parameters, signature):
    values = sample_arguments(signature)
    expected = {name: value.copy() for name, value in values.items()}
    expected["out"][:] = values["x"] ** 2
    return ValidationCase(values, expected, rtol=1e-6, atol=1e-7)
```

In `definition.py`, declare writes, qualification and both provider bindings:

```python
from akas.kernels.contracts import Implementation, OutputABI
from akas.ops.square.validation import validation

OUTPUT_ARGUMENTS = {"reference": {"square": OutputABI((-1,), ("out",))}}
ROW_INDEPENDENT = ("square",)
VALIDATIONS = {"square": validation}
IMPLEMENTATIONS = tuple(
    Implementation(
        "square", "square.kernel", "reference",
        provider=provider, factory="square", semantic_keys=("r", "c"),
    )
    for provider in ("cuda", "webgpu")
)
```

Only declare row independence when each output row depends on its corresponding
input row and declared row-local state. It permits graph pruning; it is not a
general performance hint.

In both `cuda.py` and `webgpu.py`, bind the same computation and expose each
provider's approved inventory:

```python
from .kernel import square
from .definition import IMPLEMENTATIONS as _CHOICES

_PROVIDER = __name__.rsplit(".", 1)[-1]
IMPLEMENTATIONS = tuple(choice for choice in _CHOICES if choice.provider == _PROVIDER)
```

These are real shared bindings. If specialization is needed later, replace or
add a selected backend implementation while keeping the same semantic cases.
Construction uses the existing public entry:

```python
from akas.ops import Capabilities, KernelSpec, define

spec = KernelSpec("square", {"r": 2, "c": 17})
cuda_kernel = define(spec, capabilities=Capabilities("cuda"))
webgpu_kernel = define(spec, capabilities=Capabilities("webgpu"))
```

This constructs compiler IR, not device executables. The
[source selection](../src/akas/compilation/selection.py) module's
`source(...).render()` exports a selected callable;
[artifact emission](../src/akas/compilation/emit.py) builds it. The
[WebGPU tests](../tests/test_webgpu_model_lowering.py) demonstrate source export,
artifact building, device loading and launching with buffers. Add equivalent
execution cases for your operation before declaring support. Use a fresh process
or clear registry caches after creating a family in a running Python session.
No example `square` package is installed by this guide.

## Extend a complex operation

Define the same callable inputs, results and state effects before choosing a
backend. Give multi-launch algorithms an explicit intermediate contract: split
attention returns partial values and normalization statistics, then a merge
produces the output. A GDN verification plan distinguishes tentative results from
accepted persistent state. One operation callable may therefore build several
device callables without losing the function-input/result format.

Resolve quantization storage, metadata and inline access through the format
interface. GGUF identifiers, byte/word packing, scales and rounding rules belong
in [quantization](../src/akas/quantization/README.md). Packed lookup and portable packed projection use a quantization-owned
weight reader. Accelerated format-specific consumers and metadata ABIs still
need explicit compatible implementations. Do not insert an unconditional
full dequantization pass to make a shared kernel look simpler.

Before adding a backend specialization, name the concrete reason in its local
README: unsupported primitive, necessary physical layout, different launch plan,
or a benchmarked benefit. Reuse the common arithmetic or inline helpers whenever
that preserves the optimized algorithm. Keep target/geometry eligibility local
and preserve compiler source-export dependencies and tuning declarations.

## Validate support and preserve optimization

Both providers must pass the same semantic cases, including supported dtypes,
shapes, masks, tails, aliases and state transitions. Stateful tests must verify
unselected requests stay unchanged. GDN verification also needs zero, partial and
full acceptance. Codec tests must check encoded values and metadata, not only a
decoded approximation. Specialized reduction orders may require stated numerical
tolerances; do not erase intentional rounding differences between contracts.

Source construction, device compilation, device execution, and performance are
separate checks. Retain current eligible optimized selections and compare whole
plans, including conversion and merge launches, as well as individual kernel
timings. Run performance measurements without competing GPU work and state which
device/target was tested. A shared source is not automatically slow, and a backend
directory is not evidence of an optimization gain.

For existing examples, these commands exercise the declaration contracts and the
two device test paths:

```sh
uv run pytest tests/test_kernel_definition.py tests/test_operation_writes.py
TENSOR_WEBGPU=1 uv run pytest tests/test_webgpu_model_lowering.py
TENSOR_QWEN_CUDA=1 uv run pytest tests/test_qwen_kernels.py
```

The device suites require the matching adapter/compiler setup. Without those
environment variables, their native cases skip. See the
[compilation guide](../src/akas/compilation/README.md) for build setup. A new
operation needs its own equivalent execution cases on both providers; running
existing families' tests does not validate the new operation automatically.

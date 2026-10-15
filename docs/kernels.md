# Tensor inputs to explicit results

Every standalone device definition in `ops` and `quantization` uses the same
`@kernel` boundary. The surrounding factory specializes geometry and scheduling;
the computation has typed tensor inputs, result declarations, a device body and
an explicit return.

```python
from akas.kernels.kernel import Tensor, kernel, output


def linear(spec, schedule):
    import tilelang.language as T

    rows, depth, columns = (spec.parameters[k] for k in ("r", "k", "o"))

    @kernel
    def compute(
        x: Tensor((rows, depth), "bfloat16"),
        weight: Tensor((columns, depth), "bfloat16"),
    ) -> Tensor((rows, columns), "float32"):
        out = output((rows, columns), "float32")
        # The existing tiled/pipelined tensor-core computation writes out.
        ...
        return out

    return compute
```

This is a definition pattern; the abbreviated body above is not an executable
projection implementation. Read [CUDA GEMM](../src/akas/ops/linear/cuda.py)
for the complete optimized computation.

## Results and effects

- A fresh result is declared with `output(shape, dtype)` before the computation.
- Several results are returned as a tuple, such as `return out, stats`.
- Mutated state remains an input and is returned, such as `return state, out`.
- Tensor annotations accept a flat count or a multidimensional shape.
- Geometry checks and schedule choices occur in the surrounding factory.

Attention partials return values and statistics; merges take those partials as
inputs and return the merged output. Q8/Q16 conversions return encoded words,
scales and integer sums. Cache writers return their modified cache buffers.
Inline scalar decoders remain ordinary inline helpers.

## Compiler boundary

`kernels/kernel.py` validates the definition and lowers fresh result declarations to
explicit output arguments. It removes only those declarations and the result
return; the device computation remains unchanged. TileLang then processes the
resulting function through the TileLang frontend.

The returned object is a compiler primitive. This authoring notation does not
allocate device tensors or launch a kernel when a Python definition is imported.
The surrounding graph owns allocation, aliases, scratch reuse and execution.
Returning several buffers creates neither a conversion stage nor an extra launch.

An explicit physical buffer order can be selected with `output(..., position=N)`; physical
buffer names can be selected with `@kernel(names={...})`. These are physical ABI
compatibility options. Ordinary new kernels use appended output buffers and
their declared names.

The shared boundary requires annotated inputs, matching result annotations, one final
return of named buffers, and explicit fresh outputs. It rejects undeclared
results, duplicate output positions and unreturned direct stores after macro
expansion. A repository structural check also rejects bypassing this boundary
through standalone `@T.prim_func` or `primitive(...)` definitions.

## Optimized variants

CUDA and WebGPU share this definition structure, while their bodies retain the
instructions suited to their hardware. Tensor-core MMA, subgroup reductions,
integer dot products, packed loads, pipelines and fused epilogues stay inside
those bodies or their shared inline helpers.

Backend selectors retain the existing geometry and tuning decisions. Shared
projection helpers receive explicit paired/residual options; shared activation
packing receives a component count. They no longer dispatch these computations
using string operation names.

Quantization formats own their standalone signatures and storage declarations.
For example, [Q16 WebGPU](../src/akas/quantization/activations/q16/webgpu.py)
obtains all result layouts from its format and consumes the shared packing body
inline. [FP8 CUDA](../src/akas/quantization/activations/fp8/cuda.py) uses the same
tensor-result structure with its specialized encoding body.

Validate result signatures, supported geometry, numerical outputs and state
effects with independent references. Qualify compiled kernels on the target
device and measure the complete launch plan. See [operation authoring](operations.md).

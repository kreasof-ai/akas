# Quantization

A format owns its encoded representation and how to access it. An operation owns
its computation, tiling and scheduling. CUDA and WebGPU use the same codec
interface even when their physical buffers or hardware instructions differ.

## The same six files as operations

```text
<activations|weights|kv>/<format>/
    __init__.py       exports FORMAT
    definition.py     storage, geometry, providers and conversion registrations
    kernel.py         shared callable computations and inline access
    cuda.py           CUDA factories, bind, CAPABILITIES and IMPLEMENTATIONS
    webgpu.py         WebGPU factories, bind, CAPABILITIES and IMPLEMENTATIONS
    validation.py     independent numerical expectations
    README.md         numerical rules and actual backend capabilities
```

The common contracts live in [formats.py](formats.py). Multiple formats share
small arithmetic/access helpers in [_shared](_shared); this is infrastructure,
not another format or backend hierarchy. Shared integer activation arithmetic
lives in [activations/_shared](activations/_shared).

| Format | CUDA | WebGPU |
| --- | --- | --- |
| [Activation FP8](activations/fp8/README.md) | Quantize, dequantize, fused scalar/tile reads | Same capabilities; packed words and software conversion |
| [Activation Q8](activations/q8/README.md) | Quantize, dequantize, fused scalar/tile reads | Same capabilities; existing packed integer consumer retained |
| [Activation Q16](activations/q16/README.md) | Quantize, dequantize, fused scalar/tile reads | Same capabilities; existing packed integer consumer retained |
| [Weight FP8](weights/fp8/README.md) | Dequantize, fused scalar/tile reads; specialized operand widening | Dequantize, fused scalar/tile reads |
| [Weight GGML](weights/ggml/README.md) | Dequantize, fused scalar/tile reads; qualified pair/dot specializations | Dequantize, fused scalar/tile reads; retained packed GEMV/integer helpers |
| [KV FP8](kv/fp8/README.md) | Pair conversions, inline reads/writes and tile reads; qualified native pair/packed access | Pair conversions, inline reads/writes and tile reads |

These are codec capabilities. Model support, attention geometry, native FP8 matrix
instructions, and a faster implementation for every shape are separate matters.
Weight packages consume existing checkpoint encodings; they do not advertise a
new weight quantizer. Activation Q8 is different from GGML Q8_0 weights. Q16 here
means two residual Q8 planes, not an INT16 tensor.

## Storage and inline access

Use an explicit provider for physical allocation:

```python
from akas.quantization import get

format = get("activations", "q16")
storage = format.storage((4, 256), provider="cuda")
values = storage.named["values"]
scales = storage.named["scales"]
sums = storage.named["sums"]
```

`Storage.buffers` fixes the physical order: values, then metadata in
`metadata_names` order. CUDA and WebGPU can pack values differently without
changing the logical format. Omitting `provider` requests the canonical layout;
it is not a device-allocation shortcut.

Inside a computation factory, bind before declaring the device kernel:

```python
codec = format.bind((4, 256), provider="cuda")
read = codec.require("read")
load_tile = codec.require("read_tile")
```

Every scalar reader follows `read(encoded, *coordinates) -> float32`, where
`encoded` is a tuple of buffers in `codec.read_buffers` order. Matrices use
`(row, column)` coordinates. KV uses `(slot, head, position, channel)`.
The tuple and Python dispatch disappear during macro expansion.

Matrix tile access follows
`load_tile(encoded, tile, first_row, first_column)`: the operation owns the tile,
its element type, and the schedule. Readers decode each element into that tile
and zero-fill matrix boundaries. KV tile access additionally takes slot/head and
requires the caller to supply valid cache bounds.

Q8/Q16 sums are auxiliary data for packed integer dot products. Scalar/tile
reading needs only values and scales, so `read_buffers` omits sums. This avoids
unused ABI arguments. `Storage.buffers` still describes the complete encoding.

Ask `format.capabilities(provider)` before selecting a mechanism. A bound
`require("read_pair")`, `require("dot")`, or `require("write_pair")` rejects an
unsupported specialization. Availability may also depend on encoding or shape.
A scalar compatibility implementation does not establish optimized packed-dot
support. Small adapters in [_shared/access.py](_shared/access.py) retain the
existing attention ABIs while using the same tuple readers and writers.

`format.supports(capability, shape, provider=..., **options)` checks a specific
encoding and geometry without importing its compiler. A backend may refine its
capability inventory with a compiler-free `supports` function. For example,
CUDA GGML's `dot_half` requires Q4_0 and a channel count divisible by 256;
unavailable specializations keep the ordinary reader path. Unpermuted single
weight pieces retain packed capabilities through `bind_parts`.

## Fused computation

[Linear](../ops/linear/README.md) supplies an additive `quantized_linear` operation
under the `quantized` buffer contract. It accepts ordinary F32 or FP8/Q8/Q16
activations, and GGML or block-FP8 weights. Encoded metadata is explicit in its
functional arguments. For example:

```python
from akas.graph import Graph
from akas.quantization import get

provider = "webgpu"
rows, depth, columns = 4, 256, 128
storage = get("weights", "fp8").storage((columns, depth), provider=provider)
graph = Graph(contract="quantized")
x = graph.buffer("x", (rows, depth), "float32", role="input")
w = graph.buffer("w", storage.values.shape, storage.values.dtype, role="weight")
s = storage.named["scales"]
scales = graph.buffer("scales", s.shape, s.dtype, role="weight")
out = graph.buffer("out", (rows, columns), "float32", role="output")
graph(
    "quantized_linear",
    dict(r=rows, k=depth, o=columns, weight_format="fp8"),
    x,
    w,
    scales,
    out,
)
```

The consumer uses GEMV for one row and tiled GEMM for multiple rows. GEMM decodes
into reusable FP16 shared-memory tiles and accumulates in F32; CUDA uses tensor
cores, while WebGPU uses a tiled floating-point schedule. GEMV shares decoded
input chunks across output rows. Neither allocates a full dequantized operand.
FP16 staging is an explicit arithmetic policy, including its rounding/range.

Legacy packed embedding/linear ABIs accept embedded metadata only. They reject
formats needing separate scale buffers; `quantized_linear` supplies explicit
scale inputs. Further metadata buffers or a different operation need a compatible
consumer ABI, rather than an implicit allocation or conversion.

Existing tuned GGML integer, packed-dot, FP8, attention and Hopper choices retain
their existing registrations and algorithms. The new operation does not silently
replace them or enable an otherwise unsupported model/backend combination.

Checkpoint readers can describe a logical matrix with `weights.Weight(shape,
buffers, parts)`. Each part names its format, encoding and logical rows; optional
row/column orders describe checkpoint layout conversions. `weights.bind_parts`
combines their codec readers. For example, Qwen GGUF concatenates differently
encoded Q/K/V or GDN projections this way, keeping the original buffers intact.
[Linear](../ops/linear/README.md), [vocabulary](../ops/vocabulary/README.md) and
[MoE](../ops/moe/README.md) consume the descriptor and fuse decoding into their
computations. These segmented consumers currently accept embedded metadata;
formats with separate scale buffers use the explicit `quantized_linear` ABI.

## Adding a format

1. Copy the six-file roles from a format in the same scope. Export only `FORMAT`
   from `__init__.py`; discovery reads `definition.py` without a compiler.
2. Define exact block sizes, geometry, payload packing, named metadata, rounding,
   scales and provider layouts. Reject invalid rows rather than truncating blocks.
3. Implement shared readers in `kernel.py`. Each backend's `bind(shape, **options)`
   returns `Codec(storage, read, read_tile, ...)`. Metadata required by reading can
   differ from auxiliary metadata required by a packed dot product.
4. Supply standalone `quantize(spec, schedule)` and/or
   `dequantize(spec, schedule)` factories for capabilities the format actually
   provides. Return typed `@kernel` computations with explicit results. Register
   operation kinds, output effects and validation recipes in `definition.py`.
5. Preserve specialized pair/vector/dot access where it helps. Validate encoded
   bits, metadata, consuming operations and state effects; compare the complete
   launch plan before claiming a performance improvement.

There is no central format-name table. Generic consumers resolve the package's
binding; specialized consumers can add qualified implementation choices.
Dynamic codec sources participate in artifact fingerprints automatically.
Checkpoint parsing owns file structure; quantization owns dequantization and
prepacking arithmetic. Graph scratch invalidation belongs to
[compilation/quantization.py](../compilation/quantization.py).

Run `uv run pytest tests/test_quantization_codecs.py`. Set `TENSOR_QWEN_CUDA=1`
for native CUDA qualification or `AKAS_COMPILE_WEBGPU=1` for WebGPU compilation.
The latter checks generated shaders and does not claim device execution.
See [benchmarking and qualification](../../../docs/benchmarking.md).

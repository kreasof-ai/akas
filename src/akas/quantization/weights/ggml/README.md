# Weight GGML

GGML encodings describe weight bytes independently of the GGUF file container.
The encoding table belongs to `definition.py`:

| ID | Encoding | Logical values/block | Bytes/block |
| --- | --- | --- | --- |
| 0 | F32 | 1 | 4 |
| 1 | F16 | 1 | 2 |
| 2 | Q4_0 | 32 | 18 |
| 3 | Q4_1 | 32 | 20 |
| 8 | Q8_0 | 32 | 34 |
| 12 | Q4_K | 256 | 144 |
| 13 | Q5_K | 256 | 176 |
| 14 | Q6_K | 256 | 210 |

Rows must contain whole blocks. Scales and other metadata are embedded in these
blocks, so `Storage.metadata` is empty. CUDA stores quantized bytes; WebGPU stores
U32 words and zero-pads a final partial word. F32/F16 remain native float buffers.
Shared `kernel.decode_host` and `kernel.prepack_host` own numerical host
conversion; the GGUF reader delegates to them and keeps its public parse errors.

## Why share and specialize

Share scalar decode mathematics and tile access across both providers. CUDA
owns native half/word loads, packed-pair decoding and lane dot products.
WebGPU owns packed word access, GEMV lane products, integer operand readers and
the retained Q8×Q4 packed-dot launch. Operation code owns the consuming schedules
and epilogues; it no longer owns these block bitfields and scale offsets.

Both providers supply `dequantize`, `read`, and `read_tile`; standalone kind is
`dequantize_ggml`. CUDA's bound `read_pair` and `dot` remain conditional on the
encoding. Missing specializations raise through `Codec.require`; scalar/tile
access remains available. Adding a codec does not replace any qualified packed
consumer with its scalar fallback.

CUDA Q4_0 supplies `dot_half(encoded, x, offset, tile, lane, row, previous)`. A logical
32-lane warp consumes 256 channels: `offset` selects an activation row from a
flat view, and each lane returns its F32 partial dot after FP16 operand rounding.
The codec carries `previous` through continuous F32 additions; the consumer owns
the warp reduction. A word view and funnel
shift read the original alternating halfword-aligned blocks; they copy no
weights and preserve the GGUF storage size. This capability is distinct from
the F32-operand `dot`, and is checked by encoding and geometry before selection.
Adjacent FP16 pairs decode coefficients before the continuous F32 lane sum;
this changes neither the operands' rounded values nor their addition order.

For aligned FP16 matrix tiles, Q4_0 `read_tile` unpacks eight coefficients from
word loads and computes adjacent pairs with FP16 arithmetic. The resulting
coefficient bits match scalar F32 decoding followed by FP16 rounding, including
signed zero and subnormal scales. Other tile precisions and unaligned columns
keep scalar decoding. Single unpermuted weight pieces retain this tile reader;
segmented or permuted layouts keep the composed reader. Operations request tile
access without interpreting the encoding or allocating a decoded weight cache.
Word views use `T.meta_var` so a macro-local alias cannot rename two distinct
input pointers to the same name in a fused kernel.

Q4_1 and Q5_K supply host, scalar and tile decoding on both providers. They have no native
packed-pair or lane-dot specialization yet. Fused projection/expert consumers use
the tile reader and keep the original encoded payload resident.

The derived Q4_0 integer cache stores signed byte words plus an F32 scale.
`definition.integer_layout` and `webgpu.integer_rhs` own that layout and access.
Prepacking preserves values but increases resident storage; it is a distinct
optimization choice.

## File roles

| File | Responsibility |
| --- | --- |
| [__init__.py](__init__.py) | Public `FORMAT` export. |
| [definition.py](definition.py) | Storage, geometry, providers and conversion registrations. |
| [kernel.py](kernel.py) | Shared callable computation and inline access. |
| [cuda.py](cuda.py) | CUDA factories, binding and capability inventory. |
| [webgpu.py](webgpu.py) | WebGPU factories, binding and capability inventory. |
| [validation.py](validation.py) | Independent numerical expectations. |

## Extending and validating

Keep block math and physical layout inside this package or shared codec helpers.
Add backend specializations behind the same binding; declare actual capabilities
and independent validation recipes. Check payload bits, scales, geometry and a
consuming operation, including the full conversion/consumer launch plan.

See the [quantization guide](../../README.md) for callable contracts, fused linear
usage, adding a format and qualification commands.

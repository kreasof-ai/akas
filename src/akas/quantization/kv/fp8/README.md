# KV FP8

E4M3 cache values with one F32 scale per up-to-128-channel block. CUDA stores
bytes; WebGPU stores U32 words. Scales keep logical block shape on both providers.
A binding uses `(slots, heads, capacity, channels)` and scalar reads take
`(slot, head, position, channel)`. Partial channel blocks are supported by the
portable path.

Both providers provide `quantize`, `dequantize`, `read`, `read_tile`, and `write`.
`kv_encode`/`kv_decode` are standalone pair conversions. The encode contract
rounds source projections to BF16 before encoding; decoding returns BF16 precision
(CUDA BF16 buffers, WebGPU F32 buffers containing BF16-rounded values). Inactive
rows retain their existing cache payloads and scales.

`kv_stage(kc, vc, ks, vs, lengths) -> (keys, values, scales)` materializes a
temporary native FP16 cache. Values before each request's length decode with
the original F32 scales and round once to FP16; later positions are zero.
The returned native-cache metadata is one, so K and V may share it. Persistent
FP8 data and its metadata are read only. Attention supplies the valid prefix;
the codec owns decoding and rounding. CUDA uses native FP8 conversions, while
WebGPU uses the shared packed-word reader behind the same callable contract.

## Why share and specialize

Share the portable cache writer, decoder and exact scaling arithmetic. The
portable `write` capability accepts the F32 projections used by reference
attention and keeps that path's existing precision policy.
CUDA additionally exposes `write_pair` for its prepared-key/native writer and
`read_packed` plus `packed_dtype` for qualified attention loads. Its native pair
writer requires a multiple of 128 channels. Other dimensions, including 64,
use portable standalone encoding. `read_tile` exists on both providers; its
caller owns valid cache bounds and the attention schedule.

`bind(shape, access={...})` is called through `FORMAT.bind`: `chunk` controls
logical-row to cache-slot mapping and `value_offset` selects a native projection
slice. These access settings do not change storage layout. A missing packed
capability does not imply the format cannot be read through the common path.

Attention owns normalization, rotary transforms, masks, positions and reductions.
Codec support alone does not make every attention geometry/model portable.

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

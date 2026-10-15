# Activation FP8

E4M3 values with one F32 scale per 128 channels. Channels must be divisible
by 128. The scale is `max(max(abs(block)), 1e-12) / 448`; encoding saturates finite
magnitudes and rounds to nearest, ties to even. Inputs follow BF16 precision.
CUDA takes BF16 input buffers; WebGPU takes F32 buffers and explicitly rounds
inputs to BF16 before encoding.

CUDA stores byte payloads; WebGPU packs four bytes per U32. Both provide
`quantize`, `dequantize`, `read`, and `read_tile`. Scalar decoding returns F32;
the consuming tile's dtype chooses staging precision. Rank-three activation
shapes flatten their leading dimensions for logical matrix reads.

## Why share and specialize

Share E4M3 mathematics and the matrix access contract. CUDA uses native conversion
helpers; WebGPU needs word packing and deterministic software scaling/encoding.
Its encoder therefore lives in `webgpu.py`. These differences do not require
different consumer APIs. Existing CUDA conversion bodies remain intact.

Standalone kinds are `quantize` (block-fp8 contract) and `dequantize_fp8`.
This representation differs from weight FP8's 128×128 scaling and metadata type.

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

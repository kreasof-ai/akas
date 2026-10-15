# Weight FP8

Finite E4M3 weights with one BF16 scale per 128×128 weight block. Channels
must be divisible by 128; the final row block may be partial. Matrix and expert
(rank-three) payloads keep logical shape on CUDA using bytes. WebGPU packs four
bytes into each U32 and holds the same BF16-rounded scales in F32 buffers.

Both providers implement `dequantize`, `read`, and `read_tile`. The standalone
kind `dequantize_weight_fp8` applies scales and returns F32 values. CUDA's retained
`e4m3_decode` factory is a raw-byte qualification probe; it is not the full weight
dequantizer. CUDA also retains its specialized operand widening/permutation and
conversion source used by existing accelerated expert/projection kernels.

## Why share and specialize

Share logical indexing and scaling. CUDA uses native byte conversion; WebGPU
uses software decoding of packed words. Consumers select FP16 staging explicitly
and can reuse decoded tiles without creating a full FP16 weight matrix.
The generic `quantized_linear` consumer supports both backends, independently of
whether a particular model's native FP8 expert/attention kernels are available.

For host preparation, `kernel.prepare_host(bits, scales, provider=...)` accepts
logical U8 E4M3 payloads and numerical scales, preserves BF16 scale precision,
and returns named arrays matching `FORMAT.storage`. Upload each array using its
declared storage dtype. No GPU weight quantizer is advertised.

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

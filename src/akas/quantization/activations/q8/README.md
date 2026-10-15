# Activation Q8

One signed-byte plane with blocks of 32 values. Each block has an F32 scale
and an I32 correction sum. The scale is `max(max(abs(block))/127, 1e-20)`;
quantization rounds to nearest, ties to even, without Q16's initial FP16 rounding.
Payload bytes pack into U32 words on both providers.

Both CUDA and WebGPU provide `quantize`, `dequantize`, `read`, and `read_tile`.
Conversion supports multiple rows; the earlier WebGPU single-row restriction is
removed. Standalone kinds are `quantize_q8` and `dequantize_q8`.

## Why share and specialize

The shared block reduction and signed-byte packing need no WebGPU-specific
instruction. CUDA binds that shared conversion. Scalar/tile readers consume
values and scales; optimized integer consumers additionally use the sums.
The retained packed WebGPU dot specialization requires one row and a depth
divisible by 128; it rejects other geometries. The common `quantized_linear`
consumer supports the format's other valid matrix shapes.
The retained WebGPU Q8×Q4 packed-dot implementation is owned by
[GGML](../../weights/ggml/README.md), and CUDA can consume Q8 through the fused
FP16-staging linear path. A native CUDA packed integer specialization remains a
separate optimization, not a prerequisite for format compatibility.

This is activation Q8, not GGML Q8_0 checkpoint weights.

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

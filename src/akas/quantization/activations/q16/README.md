# Activation Q16

Two residual signed-byte planes, not an INT16 representation. Inputs first
round to FP16. Each 32-value block encodes a Q8 plane, subtracts its reconstruction,
and encodes the residual. Each plane stores F32 scales and I32 correction sums;
payloads, scales and sums use plane-major order.

Normally each plane gets its own maximum-based scale. Explicit
`fixed_residual=True` uses the first scale divided by 254 for the second plane;
it changes numerical policy and is not an interchangeable tuning choice.

Both CUDA and WebGPU provide `quantize`, `dequantize`, `read`, and `read_tile`.
Standalone kinds are `quantize_q16` and `dequantize_q16`. Fused readers reconstruct
both planes and omit correction sums from their read operands.

## Why share and specialize

Share the residual mathematics, shared-memory reduction, packing and readers.
WebGPU retains its optional subgroup reduction/shuffle path and shared-memory
fallback. CUDA selects the shared-memory path and emits no WGSL subgroup calls.
Its fused linear compatibility path stages decoded tiles in FP16 for tensor-core
computation. Existing WebGPU packed integer matrix consumers remain selectable;
a CUDA integer-dot specialization can be added behind the same capability model.

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

# KV quantization

Each cache format follows the same six-file pattern as operations:

```text
fp8/
    __init__.py
    definition.py
    kernel.py
    cuda.py
    webgpu.py
    validation.py
    README.md
```

[codec.py](codec.py) supplies scope-specific lookup, unquantized dtype checks and
`pair_layout(name, shape, provider=...)`. Persistent state order is K values,
V values, K metadata, V metadata; each cache owns its metadata independently.
Provider selection reaches physical allocation, including packed WebGPU words.
This K/V pairing convenience is not part of the generic format contract.

[FP8](fp8/README.md) provides shared scalar/tile access, writing and standalone
pair conversions on CUDA and WebGPU. Native prepared-key writing and packed
attention loads are additional CUDA capabilities with explicit geometry limits.

Attention owns masks, rotary transforms, cache positions and computation.
Quantization owns payload packing, scales, rounding and access. Add a package
with a local `FORMAT`; no central format-name table needs editing.

See the [quantization guide](../README.md) for the common binding and file roles.

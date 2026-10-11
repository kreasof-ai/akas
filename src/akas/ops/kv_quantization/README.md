# KV quantization

KV formats are independent of model definitions, attention algorithms and weight
quantization. `format.py` describes payload/metadata storage and discovers a
format's `FORMAT` definition from its named module. It imports no compiler.

```text
kv_quantization/
├── format.py          # Format/Storage contracts and lazy discovery
├── fp8.py             # Existing E4M3 block-128 layout and provider declaration
├── cuda/fp8.py        # Inline pair encoder, scalar decoder and packed decoder
├── kernels.py         # Standalone kv_encode/kv_decode operations
└── implementations.py # Register standalone operations under the kv-pair ABI
```

To add a codec, provide `<name>.py` with `FORMAT = Format(name, layout, providers)`
and a provider module. `layout(logical_shape)` returns `Storage(payload,
metadata)` using `BufferSpec`s. K and V store separate metadata; persistent-state
ordering is K payload, V payload, K metadata, V metadata. The format owns packed
storage dimensions and metadata precision. Shared memory allocation asks the
format for this layout rather than assuming FP8 scales.

The current CUDA pair ABI accepts one metadata buffer per cache. Its provider
implements `writer(head_dim, value_offset=0)`, returning an inline macro with
arguments `(keys, values, row, head, slot, position, kc, vc, ks, vs)`, and
`decoder()`, returning a macro `(cache, metadata, slot, head, position, channel)`.
Keys arrive prepared in BF16; values arrive as an FP32 projection and the existing
FP8 encoder rounds them through BF16. A codec defines its own encoding and
metadata lookup. The decoder returns a floating-point value. For compatible
packed attention loads, declare `PACKED_DTYPE` and supply `packed_decoder()`;
unsupported representations should reject that fast path. Layouts with more
metadata buffers need another compatible operation ABI and attention binding.

Attention owns normalization, rotary transforms, masks, cache positions and
reductions. `attention/cuda/quantized.py` delegates storage and conversion to the
selected codec. Existing split, workspace and Hopper paths delegate FP8 conversion
as well, while retaining their qualified FP8 layouts. Their hardware/layout
qualification is separate from format availability.

The existing `kv_dtype="fp8"` option, block size, rounding, scales and cache ABI
are preserved. Standalone encode/decode kernels exercise dimensions 128/256/512
and one/three KV heads; FP8 encoding currently requires channels divisible by
128. The fixed Qwen attention geometry and other providers' codec support are
separate work. Choosing another codec does not make an incompatible attention
layout valid automatically.

The LFM reference path adds portable `reference_writer` / `reference_decoder`
helpers over F32 projections, including 64-channel partial blocks. A descriptor
may provide a `layouts` mapping, selected by `storage(shape, provider=...)`. FP8
on WebGPU packs four payload bytes per `uint32`; scales retain logical block
shape. CUDA reference payloads use bytes. The original optimized CUDA writer
still has its qualified multiple-of-128 channel contract. Backend availability
does not make an existing attention ABI compatible with a new provider.

# Operation ownership

Read an operation's `compose.py` for its host-side bindings, then its kernel file
for arithmetic. CUDA variants live under `cuda/`; WebGPU files exist where
supported. Factories import TileLang only when constructing a kernel.

| Operation | Responsibility |
| --- | --- |
| `attention`, `gdn`, `convolution` | Mixer composition, cache and recurrent arithmetic |
| `linear` | Packed, BF16 and FP8 projections; shared fused projection implementations |
| `ffn`, `moe` | Dense/packed FFN, activations, routed and shared experts |
| `normalization`, `residual` | RMS, residual normalization and addition |
| `quantization`, `reductions` | Storage decoders, conversion, scratch validity and reductions |
| `vocabulary`, `sequence` | Embedding/head/token selection and row/position control |
| `mtp`, `speculation` | Draft joining and device control for speculative rounds |

`models/blocks.py` declares decoder order and weight roles. `execution/plan.py`
binds calls and borrows buffers and kernels. `memory/layouts.py` describes
persistent state. Execution owns graph lifetimes, state advancement, commit and
repair; generation owns prompt, termination and acceptance policy.

`compilation/factories/` routes opcodes to these implementations without embedding
arithmetic. Numerical differences remain explicit: native FP8 decode and
prequantized prefill have different operand contracts; packed weights retain
their recorded quantization type; split projections retain reduction order.
The fixed-slot FP8 profile's supported geometry is explicitly validated in
`execution/bindings/block_fp8/geometry.py`.

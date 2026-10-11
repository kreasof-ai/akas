# Minimal model definitions

Each model family has three files. `config.py` describes architecture and supported
checkpoint variants. `weights.py` validates tensor inventories and declares raw
storage and projection layouts. `model.py` declares decoder blocks and selects an
execution contract. Package initializers contain no implementation.

```text
models/
├── lfm2/
│   ├── config.py
│   ├── weights.py
│   └── model.py
└── qwen35/
    ├── config.py
    ├── weights.py
    └── model.py
```

LFM2's definition composes short convolution or attention with a packed SwiGLU
feed-forward block. Qwen's definition composes GDN or attention with dense or MoE
feed-forward blocks, selected from its configuration. Both use `ops.DecoderBlock`.
Weight prefixes, normalization names, embedding and output-head bindings are
explicit architecture data.

The model files do not implement request admission, forward loops, scratch
allocation, graph capture, cache reset, speculative acceptance or state repair.
Those live in shared components:

| Responsibility | Owner |
| --- | --- |
| Decoder block composition and mathematics | `ops/decoder.py`, `attention.py`, `gdn.py`, `shortconv.py`, `ffn.py` |
| Linear, RMS/residual norm and quantization reuse | `ops/linear.py`, `normalization.py`, `quantization.py` |
| Packed-weight independent handles | `execution/packed/` |
| Block-FP8 fixed-slot decode/prefill | `execution/hybrid/` |
| BF16 compact batches over persistent slots | `execution/batched/` |
| Plan rewrites and numerical profiles | `execution/optimizations/`, `profiles/` |
| Persistent state allocation and recurrent journals | `memory/`, `ops/hybrid_state.py` |
| Bit-preserving upload and raw storage readers | `checkpoints/` |
| Bundle coverage, loading and integrity | `artifacts/` |
| Greedy and MTP orchestration | `generation/` |
| Admission, work budgets and independent-handle adapter | `scheduling/` |

Executors receive checkpoint and architecture definitions from the model classes.
Decoder blocks dispatch to operation implementations appropriate to their numerical
contract. The current packed, block-FP8 and BF16 paths retain their existing geometry,
precision and qualification limits; adding a model does not make incompatible kernel
layouts interchangeable.

Start with `lfm2/model.py` or `qwen35/model.py`, follow `DecoderBlock.compose`, then
read the corresponding execution path. Public `akas` entry points remain available.

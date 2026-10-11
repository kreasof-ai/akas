# Minimal model definitions

Each model family has three files. `config.py` describes architecture and supported
checkpoint variants. `weights.py` validates tensor inventories and declares raw
storage and projection layouts. `model.py` declares decoder blocks and selects a
numerical binding. Package initializers contain no implementation.

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
feed-forward blocks, selected from its configuration. Both use `models.blocks.DecoderBlock`.
Weight prefixes, normalization names, embedding and output-head bindings are
explicit architecture data.

The model files do not implement request admission, forward loops, scratch
allocation, graph capture, cache reset, speculative acceptance or state repair.
Those live in shared components:

| Responsibility | Owner |
| --- | --- |
| Decoder block composition and mathematics | `models/blocks.py`, `ops/{attention,gdn,convolution,ffn,moe}/` |
| Linear, RMS/residual norm and quantization reuse | `ops/{linear,normalization,quantization}/` |
| Request admission, validation, prefill, decode and teardown | `execution/executor.py` |
| Packed, block-FP8 and BF16 buffer/kernel bindings | `execution/bindings/` |
| Plan rewrites | `execution/bindings/block_fp8/optimizations/` |
| Measured operation schedules | `ops/schedules/`, `ops/linear/schedule.py` |
| Persistent state allocation and recurrent journals | `memory/`, `execution/bindings/bf16/journal.py` |
| Bit-preserving upload and raw storage readers | `checkpoints/` |
| Bundle coverage, loading and integrity | `artifacts/` |
| Prompt, termination, acceptance and output lookup | `generation/` |
| Device stepping and MTP orchestration | `execution/` |
| Queue admission and work budgets | `scheduling/` |

Numerical bindings receive checkpoint and architecture definitions from the model classes.
Decoder blocks dispatch to operation implementations appropriate to their numerical
contract. The current packed, block-FP8 and BF16 paths retain their existing geometry,
precision and qualification limits; adding a model does not make incompatible kernel
layouts interchangeable.

Start with `lfm2/model.py` or `qwen35/model.py`, follow `DecoderBlock.compose`, then
read the corresponding execution path. Public `akas` entry points remain available.

For a new composition that uses existing operation ABIs, start with
[the three-file example](../../../examples/models/residual/) and
[the shared forward-plan contract](../planning/README.md). Its compiler and runtime
derive their work from the same calls, without another model-specific recipe.

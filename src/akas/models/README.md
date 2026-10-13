# Model definitions

A model family owns three files: `config.py` describes architecture,
`weights.py` maps and validates checkpoint tensors, and `model.py` composes
operation graphs. Initializers expose public names. Shared optional heads live
under `models/heads/`; MTP implements the common speculative proposer protocol.

| Responsibility | Files |
| --- | --- |
| Model declaration and compiled-program construction | `models/definition.py` |
| Architecture and checkpoint mapping | `models/{lfm2,qwen35}/{config,weights,model}.py` |
| Operation composition and qualified kernel choices | `ops/*/operation.py`, `ops/contracts.py` |
| Semantic calls, buffers and request bindings | `graph.py` |
| Geometry compilation and default beam search | `compilation/compile.py`, `compilation/tuning.py` |
| Shared request lifecycle and phase execution | `execution/executor.py`, `execution/batch.py` |
| State, scratch and device replay | `execution/{state,workspace,replay}.py` |
| Speculative acceptance and proposer orchestration | `generation/speculative/`, `models/heads/mtp.py` |
| Checkpoint formats and storage partitions | `checkpoints/` |
| Activation, weight and KV codecs | `quantization/{activations,weights,kv}/` |

The common compiler reads graph requirements and qualifies operation-owned
implementations by default. A model that composes existing operations does not
need its own compiler, tuner or executor. Supported providers and storage layouts
are explicit operation capabilities; unsupported combinations must fail clearly.

Dense Qwen, MoE Qwen and LFM constructors all return the same Executor. LFM
supports GGUF and safetensors through its checkpoint declarations; packed and
floating projections share graph composition. Model/provider compatibility remains
explicit: a definition does not silently make unsupported layouts interchangeable.
Legacy numerical bindings and model-specific compiler recipes have been removed.

Start with [the three-file example](../../../examples/models/residual/),
[the graph contract](../graph.py), and [the compiler map](../compilation/README.md).

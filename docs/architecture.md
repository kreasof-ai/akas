# Engine architecture

Model definitions supply architecture, weight mappings and numerical constraints.
Shared operations compose their forward paths. Shared executors own resident
resources, request state, execution batches and graph lifetimes. Tensor supplies
compilation, buffers, artifact execution and device graphs.

```mermaid
flowchart LR
    Definition[Model config and weight layout] --> Blocks[Decoder blocks]
    Blocks --> Operations[Attention / GDN / convolution / FFN]
    Operations --> Plan[Ordered kernel plan]
    Caller[Local caller or HTTP adapter] --> Scheduler[Request scheduler]
    Scheduler --> Execution[Packed / fixed-slot / compact-batch executor]
    Execution --> Plan
    Execution --> State[Request state and scratch]
    Plan --> Tensor[Tensor binding and device execution]
```

## Models and operations

`models/lfm2/` and `models/qwen35/` each contain `config.py`, `weights.py` and
`model.py`. Dense and MoE Qwen are variants of one family. The definitions compose
`DecoderBlock` objects with explicit mixer, feed-forward and weight roles. See
[the model reading map](../src/akas/models/README.md).

`ops/attention.py`, `gdn.py`, `shortconv.py` and `ffn.py` implement the block
operations. `linear.py`, `normalization.py` and `quantization.py` compose projection,
normalization and scratch reuse. `DecoderBlock.compose` invokes normalization,
mixing and feed-forward operations through the selected numerical profile.
[Shared operations](../src/akas/ops/README.md) describe those contracts.

Packed LFM2 has independent handles with serialized shared scratch. Block-FP8
MoE uses a fixed-slot resident batch plus chunk executors. Dense BF16 uses compact
execution batches over persistent request slots. These contracts live in
`execution/packed/`, `execution/hybrid/`, and `execution/batched/`. Checkpoint types
and block definitions are injected by model entry points. Numerical differences
remain explicit; these paths do not silently change precision or rounding.

`execution/optimizations/` installs checked plan replacements. `execution/profiles/`
selects measured specializations. `execution/plan.py` resolves kernels, applies
profile parameters, orders calls and binds named arguments. `tensor/executor.py`
submits and captures those calls using Tensor.

## Ownership

| Owner | Resources | Lifetime |
| --- | --- | --- |
| Caller | Tensor device | Encloses engine lifetime |
| Resident executor | Weights and loaded kernels | Shared across requests |
| Request | Position, KV/convolution/GDN state | Admission through release |
| Execution geometry | Scratch, bound calls and captured graphs | Geometry/profile |
| Scheduler | Queues, admission and work budgets | Engine lifetime |
| Transport | Connections, events and cancellation signals | Client connection |

Plans borrow resources. Request slots remain distinct from compact execution rows.
`scheduling/independent.py` adapts the `IndependentModel` request contract, without
loading checkpoints or selecting provider kernels. Model factories load resources;
request `next_token()` methods select the appropriate greedy execution/readback.
Execution and resource destruction stay on the owner thread. Submission and
cancellation use thread-safe queues/signals.

`memory/state.py` allocates explicit buffer specifications and releases partial
allocations on failure. `ops/hybrid_state.py` derives layouts from geometry.
`memory/recurrent_journal.py` owns deferred recurrent updates. Speculative
verification commits or restores state at the accepted prefix for each request.

`checkpoints/` implements GGUF/safetensors reading and bit-preserving transfer.
Models declare inventories and projection packing layouts; upload implementations
own transfer barriers and allocation cleanup. `artifacts/` implements kernel
coverage, integrity, loading and implementation identities.

`generation/` contains greedy and MTP strategies, acceptance and output lookup.
MTP executes against supplied target/draft/verifier/repair contracts. The retained
paths have different commit semantics, represented explicitly. `serving/` handles
transport over any `Engine`. Caller-owned factories provide devices, model adapters,
tokenizers and metadata; LFM2 and Qwen dense examples use the same HTTP routes.
`text/tokenizer.py` supplies one byte-level BPE implementation to both families.
`text/formats.py` loads GGUF or local tokenizer JSON rules; `text/chat.py` owns
bounded ChatML formatting independently of tokenization.
The fixed-slot MoE benchmark server retains
its experimental lifecycle and qualification limits.

## Extension and qualification

Add a model by declaring its configuration, checkpoint layout and decoder blocks.
Reuse an execution contract when geometry, state layout, precision and updates match.
Add an operation/profile variant when those numerical contracts differ. A model
should not implement its own device launch loop or graph-resource lifecycle.

Paged caches need continuation semantics for both KV and recurrent state. Token
beam search requires state fork/reorder. The scheduled API currently supports
greedy token-ID generation bounded by output length; LFM2's convenience generator
also provides tokenizer/chat formatting and EOS stopping.

AKAS depends on a pinned published Tensor revision. Consumers need no compiler or
reference framework. Existing whole-model Qwen qualification limits remain; CPU
plan parity and local CUDA checks do not qualify Hopper or WebGPU profiles.

## Offline compilation

`compilation/build.py` handles source reuse, CUDA/WGSL builds and atomic manifests.
Execution profiles accept loaded checkpoints and declare kernel specializations;
models do not supply custom producers. Profiles select geometry
and produce artifacts for the corresponding execution contracts. See
[the compilation map](../src/akas/compilation/README.md). Kernel and serving searches live in
`benchmarks/`; setup and downloads live in `tools/`. Producer dependencies do not
enter installed runtime imports.

Bundles fingerprint model definitions, shared operations, execution, transfer and
memory code. Rebuild implementation-bound bundles after changes. Historical
reports and frozen source identities remain unchanged.

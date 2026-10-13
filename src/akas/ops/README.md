# Operations

Start with an operation's `operation.py`: it declares compatible implementations,
schedule axes and composition rules. `validation.py` contains independent arithmetic
and state references. Kernel files construct Tensor/TileLang kernels; compiler
imports occur inside factories.

| Area | Responsibility |
| --- | --- |
| `linear` | Projection, paired projection fusion and split reductions |
| `activation` | Dense, packed and expert-row SwiGLU layouts |
| `normalization`, `residual` | RMS, residual normalization and addition |
| `attention` | Query/KV preparation, cache access and attention algorithms |
| `gdn`, `convolution` | Recurrent mixing and causal history |
| `moe` | Routing, expert projection and weighted/shared combination |
| `vocabulary`, `sequence` | Embeddings, sampling and row/position controls |
| `control`, `mtp` | Speculative-round controls, token feedback and head joining |
| `indexing`, `state` | State row reset and compile-time request mapping |

Weight, activation and KV codecs belong to `quantization/`, independently of model
families. [KV formats](../quantization/kv/README.md) own storage, metadata and codec
placement; attention consumes their declared interface.

The registry selects by operation contract, target, provider, phase and geometry.
The compiler performs numerical qualification and calls Tensor beam search by
default. A schedule cannot change semantic dimensions or the call's buffer ABI.
Different prefill/decode algorithms, packed operands and partition reductions remain
explicit choices; they do not require separate model executors.

Kernel consolidation remains in progress. Existing provider directories contain
unmigrated algorithms and must not be removed before parity/performance evidence.
`compilation/factories/` also remains for old producers. New model integration
should use graph calls and the common compiler rather than adding another factory.

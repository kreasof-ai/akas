# DSpark draft model

DSpark is a sidecar paired with a target model. It has its own weights and KV cache,
and borrows the target embedding and vocabulary projection. The GGUF reader checks
the sidecar geometry and target hidden-state indices before allocating device memory.

| File | Responsibility |
| --- | --- |
| `model.py` | Compose context injection and proposal phases through ordinary operations |
| `config.py` | Validate draft geometry and target feature order |
| `weights.py` | Read native draft weights and borrow target weight views |
| `proposer.py` | Consume target features, propose blocks and refresh accepted context |
| `__init__.py` | Export the draft declaration and proposer |

`DSparkDraft.compose(..., phase="context")` projects concatenated target features,
normalizes them, and injects keys/values into each draft layer. The feature input
contains one chunk, rather than a complete prompt history.

`phase="proposal"` embeds an anchor followed by mask tokens, applies bidirectional
attention within that block, and computes the shared vocabulary projection. Each
position adds a low-rank Markov bias conditioned on the preceding sampled token.
The confidence head scores those choices. Proposal execution leaves logical draft
positions unchanged.

The common `SpeculativeDecoder` verifies `[anchor, *draft_tokens]` with the target.
Target convolution histories are journaled and commit only the accepted prefix.
`DSparkProposer.repair()` replaces temporary draft KV with the accepted target
features and advances draft context by that same count. Future rejected KV remains
outside the live prefix. Close the proposer before the target so borrowed weights
remain valid throughout execution.

Target composition must expose the sidecar's named layer features in decode,
prefill and verify phases. Intermediate prefill declares those features explicitly;
the compiler keeps them live until the chunk consumer runs. Both models use the
same generic program compiler, executor, operation families and backend selection.
An external draft currently uses uncaptured coordinator rounds; individual model
forwards can still use captured graphs. Whole-round capture requires support for
this external feature handoff and must not reuse assumptions of a shared MTP head.

The model's [official sidecar](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark-GGUF)
defines a nine-position block and five target feature layers. A full proposal can
produce up to ten target outputs after verification, including its bonus token;
actual speed depends on acceptance and the cost of both forwards. F16 cache and
weight precision, accepted tokens and timing boundaries must accompany any report.

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

Provider directories contain distinct scheduling and operand-layout algorithms.
They share the definition boundary below and independent operation qualification.
Model integration uses graph calls and the common compiler; the old family factory
package has been removed.

## Defining and selecting a kernel

`definition.py` is the common entry for every provider:

```python
from akas.ops import Capabilities, KernelSpec, define

kernel = define(
    KernelSpec("linear", {"r": 4, "k": 256, "o": 64}, contract="reference"),
    {"threads": 64},
    Capabilities(provider="webgpu", target="webgpu-portable-v1"),
)
```

Models declare graph calls, rather than calling this factory themselves. The
compiler uses the same entry for ordinary compilation and beam candidates,
with a resolved implementation declaration. Exported sources hash the selected
leaf and its imports; they do not require discovery of all operation registrations.
Semantic dimensions remain separate from schedule choices and cannot be changed
by tuning. Numerical recipes verify the complete buffer ABI and persistent state.

`operation.py` owns eligibility and supported search axes, `compose.py` owns
multi-call operation structure, and `rewrites.py` owns conservative fusions.
CUDA and WebGPU leaves implement actual algorithm alternatives behind this
boundary. Hardware-specific algorithms belong to `hardware/<architecture>/`:
Hopper projection, attention, prefill and expert kernels live in
their respective operation directories. These paths express real hardware
requirements rather than creating another model execution pipeline.

Chunk attention policies are owned by `attention/lowering.py`: split-context
partial/merge stages, a scratch BF16 view of authoritative FP8 KV, and bounded
Hopper query tiles. `attention/pipeline.py` independently qualifies each physical
stage. The compiler applies these policies to any matching graph and searches
supported kernel schedules; executors allocate their scratch from the resulting
buffer declarations. No runtime graph installer is required.


Activation packing registrations and references live in `quantization/activations/`.
Weight storage helpers and FP8 operand representations live in
`quantization/weights/`; operations consume them through explicit codecs.
Operation-specific buffer output locations are declared as `OUTPUT_ARGUMENTS`, so
compiler graph passes do not infer write locations from model names.

Default beam search qualifies supported schedules for every compiled operation
geometry. Physical multi-stage pipelines are selected through explicit operation
policies; automatic search over different graph pipelines is a separate extension.

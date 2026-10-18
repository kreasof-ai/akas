# Model definitions

Every family exposes one public composition function in `model.py`:

```python
def compose(
    checkpoint,
    *,
    rows,
    chunk,
    max_requests,
    max_context,
    provider="cuda",
    phase=None,
    **model_options,
):
    ...
    return Composition(graph, commit=None)
```

`Composition` contains the execution graph and an optional accepted-prefix commit
graph. Composition is pure: it records buffers, operations and request bindings;
it opens no device and performs no compilation.

A declaration selects a reader and defaults using the same style in every family:

```python
from akas.models.definition import Definition


class MyModel(Definition):
    checkpoint_type = Checkpoint
    model_defaults = {}
```

`Definition.compose` calls the family's module function, applies the declaration's
defaults, and validates the result. There is no second `.definition` protocol.
Calling `MyModel(checkpoint, bundle, device)` opens the common Executor.

Use `<Family><Variant>` for model declarations, with matching `Config` and
`Checkpoint` suffixes. Dense and MoE are architecture variants within a family:

| Declaration (`model.py`) | Configuration (`config.py`) | Reader (`weights.py`) |
| --- | --- | --- |
| `Qwen35Dense` | `Qwen35DenseConfig` | `Qwen35DenseCheckpoint` |
| `Qwen35MoE` | `Qwen35MoEConfig` | `Qwen35MoECheckpoint` |
| `LFM2MoE` | `LFM2MoEConfig` | `LFM2MoECheckpoint` |

`DenseEngine`, `Qwen35Batch`, `DenseConfig`, `Qwen35Config`, `DenseCheckpoint`
and `Qwen35Checkpoint` remain compatibility aliases in their original modules;
new code should use the explicit variant names. `LFM2` selects dense LFM2;
`LFM2MoE` selects `lfm2moe` GGUF checkpoints. They share the family composition
and executor, while the reader validates each variant's tensor inventory.
Vision composition and multimodal naming
remain future work; these Qwen declarations compose the text model.

| File | Responsibility |
| --- | --- |
| `config.py` | Architecture and valid geometry |
| `weights.py` | Checkpoint reading, validation, storage mapping and upload |
| `model.py` | Public `compose` and model declarations |
| `_composition.py` | Optional private layer builders for larger architectures |
| `__init__.py` | Family-owned public exports |

Use explicit family imports, such as `from akas.models.lfm2 import LFM2`. Adding a
family needs no edit to `akas/__init__.py`. Historical root model imports are
resolved from each family's exports and emit a deprecation warning.

Start with [the complete bigram example](../../../examples/models/bigram/), then
read LFM and Qwen's `model.py` side by side. Both use the same entry and result.
The [residual graph example](../../../examples/graphs/residual/) demonstrates
standalone numerical graph composition without a token request lifecycle.

Operations own kernels, precision contracts and physical lowering. The common
compiler qualifies and tunes them. The common executor owns request slots,
phase geometry, state and replay. A new model using existing operations adds no
compiler, tuner or executor.

The current MTP implementation is Qwen-specific and lives in `qwen35/mtp.py`.
`generation/speculative/` retains its existing proposer orchestration. This
iteration introduces no general speculative-head framework.

## LFM2 MoE checkpoints

`from akas.models.lfm2 import LFM2MoE` selects the packed GGUF reader for LFM2 MoE.
The configuration declares leading dense layers, expert dimensions and top-k
routing. Selection uses sigmoid probabilities plus learned expert biases;
combination uses the unbiased probabilities, normalized with an epsilon of 1e-6.
There is no shared expert branch. Router and combination computations are shared
across CUDA and WebGPU. Expert coefficients remain encoded and decode inside
the consuming projection. CUDA keeps its GEMV decode and grouped tensor-core
prefill specialization. WebGPU shares decoded weight tiles across grouped token
routes and uses subgroup reductions for decode, retaining portable fallbacks.
The model declares larger prefill chunk candidates so the optimizer can amortize
the MoE bank's weight traffic across tokens.

The QAD Q4_0 checkpoint can contain other encodings for particular tensors,
including the tied embedding/output matrix. The reader preserves each tensor's
actual encoding. KV values requested as FP16 currently occupy FP32 buffers on
WebGPU after FP16 rounding; report physical allocation separately from numerical
precision. Check the [benchmark guide](../../../docs/benchmarking.md) for the
single-user long-context workload and memory measurements.

GGUF upload releases mapped host pages after each tensor transfer. WebGPU uploads
also submit queued writes in bounded batches; pending staging is limited by the
batch budget plus the largest tensor. The checkpoint is never expanded into a
full floating expert bank for upload or numerical qualification.

## Qwen checkpoint formats

`Qwen35DenseCheckpoint` and `Qwen35MoECheckpoint` accept a safetensors directory
or a GGUF file. The same `Qwen35Dense` / `Qwen35MoE` declarations compile and run
either source. `weights.py` selects the reader; private `_gguf.py` maps GGUF
names and auxiliary values into the family's logical weights. `config.py` owns
architecture metadata parsing. Adding a format does not add another model class.

GGUF projection and expert matrices stay encoded. Mixed encodings in a combined
QKV, GDN or gate/up projection are represented as logical weight pieces; codecs
decode them inside consuming kernels. Grouped-head permutations are also resolved
at reads. Norm offsets and negative SSM A values are restored by the reader;
small auxiliary tensors are converted to the existing BF16/F32 contract.

```python
from akas.models.qwen35 import Qwen35Dense, Qwen35MoE
from akas.compilation import compile_checkpoint

compile_checkpoint(
    Qwen35Dense,
    "Qwen3.5-0.8B-Q4_K_M.gguf",
    "build/qwen",
    geometries=[(1, 1), (1, 32)],
    max_requests=1,
    max_context=128,
    provider="cuda",
    target="sm_89",
)
```

Use `Qwen35MoE` for `qwen35moe` GGUFs. Dense and MoE architecture mismatches,
missing tensors and unsupported encodings fail before upload. Supported GGML
encodings are listed in the [codec guide](../quantization/weights/ggml/README.md).
Native Qwen text execution currently requires CUDA; reading GGUF does not add
WebGPU model execution, vision or GGUF MTP support. Decode/prefill qualification
still belongs to the common compiler and the actual operation implementations.

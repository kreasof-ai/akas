# AKAS

AKAS is an independent inference engine built on [Tensor](https://github.com/kreasof-ai/tensor).
AKAS owns model loading, request scheduling, persistent state, generation and serving.
Tensor owns the compiler, artifact format, buffers, device execution and GPU graph handles.
The runtime needs only AKAS, Tensor, NumPy and regex. Compiler and reference frameworks
are optional producer dependencies.

New model authors can start with the [three-file example](examples/models/residual/)
and [shared forward-plan guide](src/akas/planning/README.md). Operation registrations
own provider/hardware choices; the same plan drives compilation and runtime binding.

## Install

Use Python 3.12. The lockfile pins Tensor to a published Git revision; there is no
workspace or relative-path dependency on the Tensor repository.

```sh
uv sync --locked --extra compiler
uv run --no-sync python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
```

For development against another Tensor checkout:

```sh
uv pip install --python .venv/bin/python --no-deps -e /path/to/tensor
```

Set `TENSOR_NVRTC_HOME` to an existing NVRTC installation when available. Compiler
packages are needed to produce bundles, and are not needed by an installed consumer.
Optional extras are `webgpu`, `serving`, `reference`, and `cloud`.

## Start with a small executable example

From the AKAS checkout, create a synthetic two-layer LFM2 checkpoint and compile
its CUDA kernels for your GPU. `sm_89` is the L40S target used for local validation.

```sh
uv run --no-sync python -m benchmarks.lfm2.diagnostic --out build/diagnostic.gguf
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" uv run --no-sync python -m akas compile \
  --definition akas.models.lfm2.model:LFM2 --checkpoint build/diagnostic.gguf \
  --profile packed --out build/diagnostic --config '{"context":384,"target":"sm_89"}'
uv run --no-sync python -m akas inspect --model build/diagnostic.gguf
uv run --no-sync python examples/generate.py \
  --model build/diagnostic.gguf --bundle build/diagnostic --prompt Hello --max-tokens 8
```

The synthetic checkpoint checks contracts and execution, and produces meaningless
text. For real LFM2 checkpoints, use `benchmarks.lfm2.download` and the build tool with
the corresponding context and device target. The retained [model profiles](docs/legacy-inference.md)
describe supported formats and measured scope.

```python
import tensor
from akas import Engine

with tensor.Device() as device:
    with Engine.from_lfm2(checkpoint, bundle, device, capacity=2) as engine:
        request = engine.submit(prompt_token_ids, max_tokens=32)
        while engine.step():
            engine.events.clear()  # Consume events here when streaming.
        output_token_ids = request.output
```

`Engine` coordinates the same scheduler used by the dense serving adapter. It
supports greedy token-ID generation bounded by output length. The existing
`LFM2.generate()` convenience API additionally supports tokenizer/chat formatting
and EOS stopping. LFM2 schedules independent handles serially; it does not claim
native batching. `Engine.from_independent(model, capacity=...)` accepts any model
with `new_request()`, `reset()`, `next_token(tokens)` and `close()`; scheduling
does not inspect model names, artifact formats or providers.
`Engine.from_qwen35_dense()` uses native CUDA execution batches.
The fixed-slot Qwen MoE runner retains its specialized serving adapter and
qualification status.

## Serve an engine

The HTTP layer accepts a factory context manager yielding `Service(engine, encode,
metadata)`. The factory owns checkpoint, device, tokenizer and generation setup;
HTTP uses the same engine interface for every model. From the checkout:

```sh
uv run --extra serving python -m akas serve \
  --factory examples.serving:lfm2 \
  --config '{"checkpoint":"build/diagnostic.gguf","bundle":"build/diagnostic","capacity":2}'
```

Use `examples.serving:qwen35_dense` with a dense checkpoint and bundle for Qwen,
including its optional MTP settings. Installed callers can use `akas serve` with
their own importable factory. [The examples](examples/serving.py) show both setups.
The routes are `/health`, `/server_info`, `/tokenize`, and streaming `/generate`.
Generation accepts `input_ids` and `sampling_params` with `temperature: 0`,
`ignore_eos: true`, and `max_new_tokens`; it stops at the requested length.
Token-ID-only factories can omit the encoder.

Both examples use AKAS's shared byte-level BPE tokenizer. `Tokenizer.from_gguf()`
reads GGUF metadata; `Tokenizer.from_pretrained()` reads a local `tokenizer.json`
and optional `tokenizer_config.json`. Both return plain ID lists through
`encode()` and text through `decode()`. Format rules control splitting, Unicode
normalization, merges, added tokens and BOS/EOS. Runtime tokenization does not
import Transformers or Hugging Face tokenizers. Unsupported rules raise an error;
chat formatting remains a separate bounded ChatML helper.


## Where to read

```text
src/akas/
├── engine.py, config.py, requests.py
├── scheduling/          # Admission and prefill/decode policy
├── execution/           # One request/batch executor and numerical bindings
├── memory/              # Allocate explicit state layouts; clean up failures
├── generation/          # Prompt/termination, acceptance and output lookup
├── models/              # Three files per family: config, weights and composition
├── ops/                 # Composition and kernels grouped by mathematical operation
├── tensor/              # ABI binding, provider submission and graph handles
├── compilation/         # Offline recipes, factory routing and shared builds
├── checkpoints/         # GGUF and safetensors storage readers
├── artifacts/           # Integrity, loading and shared fingerprints
├── text/                # Tokenization and chat formatting
└── serving/             # HTTP transport over any Engine

tools/                   # General dependency setup
examples/                # Small caller-side examples
benchmarks/              # Tuning, workloads, profiles, reference checks and replay
tests/                   # CPU contracts and opt-in hardware checks
docs/                    # Architecture and retained research
```

Start with [the engine](src/akas/engine.py), [the scheduler](src/akas/scheduling/scheduler.py),
and [the shared executor](src/akas/execution/executor.py). Offline builds are described
in [the compilation map](src/akas/compilation/README.md). Then read
[LFM2 composition](src/akas/models/lfm2/model.py) or
[Qwen composition](src/akas/models/qwen35/model.py).
[Shared operations](src/akas/ops/README.md) implement attention, GDN, convolution,
feed-forward, normalization and quantized scratch reuse. Models declare block order,
weight roles and numerical constraints. [The model reading map](src/akas/models/README.md)
shows how the three-file definitions connect to shared execution. See
[architecture and extension boundaries](docs/architecture.md).

The historical [Tensor LLM migration record](docs/research/tensor-llm-migration.md)
is retained with the research reports.

Kernel and serving-profile beam searches call Tensor's `tensor.compiler.search`;
AKAS owns model-specific candidates and measurements. See
[the tuning tools](benchmarks/qwen35/kernel_search.py). Token beam search is an unimplemented
generation strategy; adding it requires state fork/reorder semantics for both KV
and recurrent caches.

## Checks and qualified scope

```sh
uv run --no-sync python -m pytest
TENSOR_LFM2_CUDA=1 TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" \
  uv run --no-sync python -m pytest tests/test_requests.py
TENSOR_QWEN_CUDA=1 TENSOR_QWEN_TARGET=sm_89 TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" \
  uv run --no-sync python -m pytest tests/test_qwen_kernels.py
uv build
```

Install the `reference` extra for hardware tests that compare against PyTorch.
Hardware tests default to skipped. WebGPU and Hopper require their own adapters
and validation; local CUDA results do not establish those profiles.

Existing LFM2 formats and Qwen experimental numerics are retained. Qwen whole-model
qualification remains open; previous FP8 KV/speculative quality failures remain
failures. Paged caches, token beam search and distributed execution are future
extensions. Moving code does not qualify a new numerical or performance profile.
Bundles fingerprint their implementations and must be rebuilt after migration.
Historical reports and frozen sources retain their original identities.

For a complete model reference, see [LFM2](examples/lfm2/README.md).
Use [`akas bench`](docs/guides/performance-harness.md) for repeated native
prefill/decode measurements, short/long context matrices, and matched baselines.

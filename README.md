# AKAS

AKAS is an inference engine built on [Tensor](https://github.com/kreasof-ai/tensor).
It owns model definitions, checkpoint loading, request scheduling, persistent state,
generation and serving. Tensor provides compilation and device execution.

Use Python 3.12. `uv.lock` pins Tensor to a published Git revision.

```sh
uv sync --locked --extra compiler
uv run --no-sync python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
```

The compiler is needed to build programs. Precompiled programs run without compiler
packages. Optional extras are `webgpu`, `serving`, `reference`, and `cloud`.

Compile a checkpoint using a model definition and the shared compiler. Kernel beam
search runs by default. Choose capacities and geometries for the workload.

```sh
akas compile --definition akas.models.lfm2.model:LFM2 \
  --checkpoint /path/to/model.gguf --out build/model \
  --config '{"geometries":[[1,1],[1,128]],"max_requests":1,"max_context":4096,"provider":"cuda","target":"sm_89"}'
akas generate --definition akas.models.lfm2.model:LFM2 \
  --model /path/to/model.gguf --bundle build/model --prompt Hello --max-tokens 32
```

Supported checkpoint formats and operation implementations depend on the model
and provider. See the [model declarations](src/akas/models/README.md).

Use `akas bench` for repeated prompt/decode measurements across batch sizes and
context lengths. `akas bench --help` lists compilation, timing, baseline comparison
and scaling regression options. `akas serve --factory module:callable` runs the
shared HTTP server with a caller-supplied service factory.

The package layout is documented beside the implementation:

- [Models](src/akas/models/README.md): architecture, checkpoint mapping and composition.
- [Planning](src/akas/planning/README.md): graph composition and optimization.
- [Operations](src/akas/ops/README.md): uniform kernel definitions and hardware choices.
- [Compilation](src/akas/compilation/README.md): program building and default tuning.
- [Execution](src/akas/execution/README.md): shared request state and phase execution.

[The three-file residual model](examples/models/residual/) demonstrates composition
using existing operations. `tests/support` contains synthetic checkpoints and
independent numerical oracles used by core tests.

```sh
uv run --no-sync python -m pytest
uv build
```

Historical experiments and reports are excluded from this repository. Daily
performance measurement lives in `src/akas/benchmarking`.

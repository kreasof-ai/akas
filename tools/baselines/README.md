# Baseline preflight

These are short execution checks for vLLM, SGLang and llama.cpp. Each worker
implements `run(config) -> report` using its own runtime. The launcher owns the
deadline, source capture and [benchmark history](../../benchmarks/README.md).
These probes do not establish optimized baseline throughput.

[Model declarations](../../benchmarks/models/) hold checkpoint paths, pinned
sources, generation modes and runtime options. Add a model by adding one JSON
file; the workers contain no model-name dispatch. Cache weights under
`build/checkpoints/<model-id>/`. A declaration may point to an existing local
cache; copying large files is unnecessary.

Create separate Python 3.12 environments under `build/baselines/vllm` and
`build/baselines/sglang`. The validated versions are vLLM 0.31.0 and SGLang 0.5.21;
install each with `uv pip install --python <environment>/bin/python`. Use a
complete CUDA toolkit and set `CUDA_HOME` and `PATH` to it. An environment with
`nvcc` but missing CUDA headers is insufficient for first-use JIT compilation.

Build llama.cpp commit `620d2225cc436ab0b001e3200e590e3b286ef9ab` under
`build/baselines/llama.cpp`, with separate `build-cuda` and `build-vulkan`
directories. Enable `GGML_CUDA` or `GGML_VULKAN` and build `llama-server`.
Use the GPU's actual CUDA architecture. Vulkan needs the loader and matching
NVIDIA graphics libraries; a software Vulkan adapter is not a GPU baseline.

Run one probe at a time from the repository root:

```sh
uv run --no-sync python -m tools.baselines vllm lfm
uv run --no-sync python -m tools.baselines vllm qwen
uv run --no-sync python -m tools.baselines vllm qwen --mtp
uv run --no-sync python -m tools.baselines sglang lfm
uv run --no-sync python -m tools.baselines sglang qwen
uv run --no-sync python -m tools.baselines sglang qwen --mtp
uv run --no-sync python -m tools.baselines llama qwen --format q4_k_m --provider vulkan
```

Native probes check B1/P32, B4/P32 and B1/P2048 with 128 total output tokens.
The llama.cpp server probe checks B1 at both prompt lengths; server concurrency
is not a controlled native batch. MTP checks compare token IDs against the
matching AR probe, which must be run first. Every probe uses one warmup and one
measured repeat, with a ten-minute process deadline by default. Cold JIT
compilation can consume several minutes; keep its cache for subsequent runs.

The launcher archives its exact model/configuration files, runtime dependencies,
checkpoint hash, raw output and log. `purpose=preflight` keeps these records
separate from final measurements. Failures and time limits remain in history.

For Qwen F16, the model declaration specifies conversion from the pinned original
safetensors checkpoint with llama.cpp `--outtype f16 --no-mtp`; a vision projector
is not the text model. The other GGUF variants reference their pinned repositories.
Use identical resulting bytes when measuring AKAS and llama.cpp.

## Measure the complete workload

Add `--optimized` to enable graph execution for the native engines, provision the
full request capacity, and measure all ten `standard-v1` shapes. Each shape runs
one warmup and three measured repetitions, generating exactly 128 tokens per
request. `--workload path.json`, `--repeats` and `--seconds` provide bounded pilot
controls. Final measurements keep their complete runtime settings and sampled
whole-device VRAM; graph capture and loading are outside inference timings.

```sh
python -m tools.baselines vllm lfm --optimized --seconds 900
python -m tools.baselines sglang qwen --optimized --seconds 900
python -m tools.baselines sglang qwen --mtp --optimized --seconds 900
python -m tools.baselines llama qwen --format q4_0 --provider vulkan \
  --optimized --seconds 900
```

The engines receive the same deterministic uniform token-ID prompts as the AKAS
optimizer: seed `17 + B*4099 + P`, using each model's declared vocabulary.
This is a synthetic workload, not an arrival trace or a language-quality test.
Prefix reuse and early EOS are disabled. The MTP run requires its matching AR
run, then compares every output ID at every shape. Completed shapes are saved
incrementally; later OOM, failure or timeout cannot erase their raw evidence.
The llama.cpp worker also disables the RAM prompt cache, idle-slot caching and
context checkpoints. These retained copies are unnecessary for empty-prefix AR
requests and can exhaust host RAM when many recurrent-model slots are active.

These measurements are concurrent request bursts. The scheduler may split or
interleave prefills; B256 does not promise a fixed GPU batch of 256 on every
step. vLLM phase rates use the maximum per-request TTFT and decode interval.
SGLang rates use scheduler prefill-completion and request-finish timestamps;
non-streaming first-output timestamps can reflect buffered outputs instead.
These include scheduler effects. llama.cpp rates use its reported server
prompt/evaluation durations, which have a separate protocol label. Preserve the
raw samples and compare like protocols; these are not interchangeable with
AKAS's isolated native kernel-phase rates. End-to-end committed output throughput
is retained for every runtime. Default optimized settings are measured settings,
not proof that exhaustive baseline tuning found the fastest possible setup.

First-use FlashInfer warmup also needs cuRAND development headers in the selected
CUDA toolkit. Keep runtime environments isolated and measurements serial.

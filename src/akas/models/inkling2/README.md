# Inkling 2 Mini

`Inkling2Mini` runs the text decoder from the official mixed NVFP4/MXFP8
checkpoint. The loader validates the original safetensors layouts and keeps
expert banks compressed in memory. The CUDA projections unpack tiles into BF16
operands and accumulate in FP32. The middle layers share weight buffers and
retain independent convolution histories and BF16 key/value caches for each
pass. Local layers keep a circular cache and global layers retain the request's
full context. Vision inputs and the DSpark draft head are outside this text
path. Single-user chat uses the checkpoint template's default thinking effort.

The [benchmark declaration](../../../../benchmarks/models/inkling-2-mini.json)
pins the checkpoint revision. Stage that snapshot at
`build/checkpoints/inkling-2-mini`. Compile the comparison program on an L40S:

```sh
uv run --no-sync akas compile \
  --definition akas.models.inkling2:Inkling2Mini \
  --checkpoint build/checkpoints/inkling-2-mini \
  --out build/inkling-2-mini/program \
  --config '{"geometries":[[1,1],[4,1],[16,1],[64,1],[1,2048],[4,2048]],"max_requests":64,"max_context":2176,"target":"sm_89"}'
```

Use the shared `concurrent-prefill-v1` workload for the L40S comparison. Its
bursts submit 1, 4, 16, 64 or 256 independent requests with 2,048 input tokens
and 128 generated tokens. Concurrent arrivals can exceed the number of
resident slots. For example, 256 resident requests at the end of this workload
would require 57 GiB for BF16 KV payloads alone.

```sh
uv run --no-sync python -m tools.baselines vllm inkling-2-mini \
  --optimized --streaming --workload concurrent-prefill-v1 \
  --seconds 3600 --history build/inkling-2-mini/evidence
uv run --no-sync python -m tools.baselines sglang inkling-2-mini \
  --runtime-source "$inkling_sglang_source" \
  --models "$inkling_sglang_models" \
  --optimized --streaming --workload concurrent-prefill-v1 \
  --seconds 3600 --history build/inkling-2-mini/evidence
uv run --no-sync python -m tools.baselines akas inkling-2-mini \
  --bundle build/inkling-2-mini/program \
  --optimized --streaming --workload concurrent-prefill-v1 \
  --seconds 3600 --history build/inkling-2-mini/evidence
```

The comparison program has 64 resident slots, decode batches of 1, 4, 16 and
64, and whole-prompt prefill for groups of 1 and 4. The runner records each
whole burst with three repetitions after one warmup. The vLLM build needs its
[Ampere and Ada support](https://github.com/vllm-project/vllm/pull/69165).
The stock SGLang build rejects the L40S during NVFP4 expert initialization.
Run `20261127085712918566-705acbf1` contains the Ada-compatible SGLang source in
`dependencies/sglang/source.tar.gz`. Point `inkling_sglang_source` to its
extracted directory and `inkling_sglang_models` to `benchmarks/models` in the
extracted AKAS `source.tar.gz` from the same run. The compatibility run uses
Marlin experts, eager prefill, and CUDA graphs for decode.

For single-request benchmarking, use `geometries` of `[[1,1],[1,2048]]` and
`max_requests` of `1` in the compile configuration. This runs the prompt in one
pass with wider expert tiles. Decode shares each BF16 cache tile across its
query heads.

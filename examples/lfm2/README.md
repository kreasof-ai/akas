# LFM2 composed execution

The model remains three files in `src/akas/models/lfm2`: architecture dimensions,
checkpoint bindings, and operation composition. `LFM2` is the existing optimized
packed-weight path. `LFM2Batched` uses the shared forward-plan compiler and runtime,
with chunked prefill and operation-owned optimized kernels. `LFM2Reference` keeps
the serial F32 baseline for numerical checks.

| Feature | Composed floating path |
| --- | --- |
| Loading | GGUF F32/F16/Q4_0/Q8_0/Q4_K/Q6_K, or single/sharded safetensors F32/F16/BF16 |
| Architecture | LFM2 short convolution, Q/K RMS norm, RoPE, causal GQA, SwiGLU, tied/separate output head |
| Providers | CUDA and WebGPU from the same composition; floating WebGPU needs shader-f16 and 32/64-wide subgroups |
| Attention cache | FP32, FP16, or FP8 with per-head block scales |
| Batches | 1, 4, 16, 64, 256; other positive sizes within memory limits |
| Sequence lengths | Live positions and lengths; decode/prefill graphs reused across lengths, including mixed prompts and padded chunk tails |
| State | Separate convolution/KV state per slot, inactive masks, full/partial reset |
| Generation | Greedy device sampling, mixed prompt lengths, independent EOS, raw prompts and supported checkpoint chat templates |
| Submission | Ordinary execution or captured/prepared graph |
| Tuning | Shared operation selection and schedule profiles; reusable performance reports |

Checkpoint geometry is validated explicitly. This implements LFM2 dense decoder
models with 64-channel heads, a three-tap bias-free short convolution, default
RoPE, and SwiGLU. It does not silently accept another architecture merely because
its name contains LFM. GGUF quantized matrices are dequantized to floating weights
once for the composed path; use `LFM2` to retain their packed representation.

Compile a local safetensors checkpoint or GGUF:

```bash
akas compile --definition akas.models.lfm2.model:LFM2Batched \
  --checkpoint /path/to/LFM2-350M --profile floating \
  --out build/lfm-reference \
  --config '{"batch":1,"capacity":4096,"provider":"cuda","kv_dtype":"fp8"}'

python -m examples.lfm2.generate --checkpoint /path/to/LFM2-350M \
  --bundle build/lfm-reference --prompt 'The capital of France is' \
  --raw --max-tokens 32 --graphs
```

Replace the provider with `webgpu` in both commands for portable shaders. Repeat
`--prompt` once per compiled batch slot. Runtime loading/generation imports no
Torch, Transformers, TileLang, TVM, or compiler. The optional oracle imports NumPy
and is deliberately kept in the example, outside runtime code.

For a small, reproducible fixture without downloading pretrained weights:

```bash
python -m benchmarks.lfm2.diagnostic --out build/lfm-fixture.gguf --context 4096
python -m examples.lfm2.checkpoint --gguf build/lfm-fixture.gguf \
  --out build/lfm-fixture-hf --dtype BF16
```

The exporter writes equivalent canonical weights under Hugging Face names and a
safetensors header/config. It produces synthetic numerical data and no tokenizer;
run it with integer-token prompts or the benchmark harness, which loads with
`tokenizer=False`. The diagnostic GGUF itself includes a small valid tokenizer.

See [daily performance measurements](../../docs/guides/performance-harness.md)
for short/long context matrices and regression checks. The numerical qualification
suites are `tests/test_lfm2_reference.py` and `tests/test_lfm2_floating.py`; enable actual execution with
`AKAS_LFM_REFERENCE_CUDA=1` or `AKAS_LFM_REFERENCE_WEBGPU=1`.

Adding another KV codec means supplying its format descriptor/provider layout and
`reference_writer`/`reference_decoder` helpers in `ops/kv_quantization`. Reference
attention currently consumes one metadata buffer per cache. WebGPU's FP8 payload
packs four bytes into a word; CUDA keeps byte buffers. These helpers are selected
through the codec interface, and quantization math stays outside the model.

Floating matrices are row-partitioned by the reusable checkpoint layout helper
(default maximum 64 MiB per binding), including large embeddings and output heads.
This lets the full 65,536-token vocabulary fit WebGPU's per-binding limit without
changing weight values. Context/batch cache allocations can still exceed adapter
limits; those failures are explicit rather than reported as performance results.

WebGPU FP16 cache values use FP16 rounding with F32 backing storage for portability;
CUDA uses physical FP16 storage. See [qualification evidence](../../docs/research/lfm-reference/README.md)
for executed cases, pretrained performance, precision policy, and adapter details.

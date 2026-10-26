# Spark X2.6 optimization assessment

NVIDIA L4, official Q4_K_M checkpoint, FP16 KV, one request, eight prefixes and 128 total outputs. Medians of three retained samples follow one warmup at each prefix. Both runtimes use identical checkpoint bytes and synthetic input IDs. AKAS uses synchronized native phases and llama.cpp uses server eval phases; timing boundaries differ.

Decode medians exceed llama.cpp at 8/8 prefixes. Every retained AKAS decode sample exceeds every retained llama.cpp sample at 8/8 prefixes. These observations describe this run and do not guarantee all workloads or future runs.

The requested 2× llama.cpp prefill target is met at 0/8 prefixes.

| Prefix | AKAS prefill tok/s | llama.cpp prefill tok/s | Prefill ratio | AKAS decode tok/s | llama.cpp decode tok/s | Decode ratio | Estimated prefill MFU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1,075.4 | 844.1 | 1.27× | 71.21 | 63.68 | 1.12× | 6.74% |
| 512 | 4,606.0 | 4,179.9 | 1.10× | 69.76 | 65.61 | 1.06× | 29.33% |
| 2,048 | 4,734.2 | 4,542.6 | 1.04× | 68.55 | 50.64 | 1.35× | 30.92% |
| 8,192 | 4,394.3 | 4,359.9 | 1.01× | 64.87 | 41.31 | 1.57× | 30.42% |
| 16,384 | 3,990.7 | 4,038.9 | 0.99× | 60.35 | 55.38 | 1.09× | 29.63% |
| 32,768 | 3,339.7 | 3,136.7 | 1.06× | 52.02 | 46.84 | 1.11× | 28.13% |
| 65,536 | 2,500.2 | 2,138.9 | 1.17× | 41.71 | 37.18 | 1.12× | 26.06% |
| 130,944 | 1,667.2 | 1,337.8 | 1.25× | 30.23 | 27.55 | 1.10× | 24.02% |

The implementation adds an optional derived FP16 layer-weight cache for prefill, larger compressed projection tiles for short prompts, shared FP32 RoPE sine/cosine tables, vectorized Q8 decode and attention tiles that share K/V loads. The FP16 cache increases device memory by about 7.55 GB. Original packed weights remain in use for decode and short prompts.

[The original comparison](https://github.com/kreasof-ai/akas/blob/ac2323d0ea0527aae1bfaded42ee80947770aefd/benchmarks/reports/20261025T235635Z-d80177ee179d/README.md) estimated prefill MFU at 4.25% for prefix 32 and 9.84–10.29% for larger prefixes. Changes against that earlier run are descriptive; the fresh llama.cpp baseline above is the matched comparison for this run.

## MFU method

[NVIDIA specifies 121 TFLOP/s dense FP16 for L4](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) (Appendix D). The sparse 242 TFLOP/s figure is not used. Estimated useful MFU is `useful_forward_FLOPs / prefill_seconds / 121e12`.

Layer projection parameters total 3,776,348,160. Dense projections contribute `2 * prefix * parameters`; the final-row vocabulary projection contributes `2 * 335544320`. Attention contributes `4 * 16 * 256 * (9 * full_pairs + 27 * window_pairs)`, with `full_pairs = N*(N+1)/2` and `window_pairs = sum(min(t,512) for t in 1..N)`.

This counts useful dense model math, excludes embedding lookups, elementwise operations, padded work, dequantization and cache/control traffic, and counts all model layers even where intermediate prefill chunks prune final-layer outputs. It is an algorithmic estimate rather than a hardware instruction counter. The generator and full per-runtime numbers are retained beside this page.

## Numerical qualification

26 checks passed, with two optional pretrained tokenizer checks skipped. CUDA checks cover independent projection and softmax references, long-position rotary values, cache wrap, inactive requests, NaN padding, ragged tails, request reorder and slot reuse against independent small-model Torch equations. Targeted compiler and quantization regression checks also passed. Full checkpoint trajectories are compared separately in the canonical benchmark report; kernel qualification does not imply identical greedy trajectories across runtimes.

The optimized path retains identical 128-token outputs to the original AKAS run at 7/8 prefixes. Divergence details and common-prefix lengths are retained beside this page. The full before/after comparison is retained in `prior-output-comparison.json`; different floating-point reduction orders need not preserve greedy trajectories.

## Reproduce

```sh
python -m tools.spark_benchmark compile --optimized --chunk 1024 --program build/spark/fast --out build/spark/compile-fast.json
python -m tools.spark_benchmark akas --chunk 1024 --program build/spark/fast --out build/spark/akas-fast.json
python -m tools.spark_benchmark llama --chunk 1024 --out build/spark/llama-fast.json
python -m tools.spark_benchmark report --runs AKAS_RUN LLAMA_RUN --out build/spark/report-fast
python -m tools.spark_mfu PATH/TO/report.json --checkpoint PATH/TO/MODEL.gguf --out build/spark/mfu.json
```

Set up CUDA libraries and TENSOR_NVRTC_HOME as described in the repository. Measurements exclude compilation, upload and phase loading. The same pinned llama.cpp revision is used as in the original comparison. GPU power, clock, temperature, utilization and memory telemetry is retained for both sequential sweeps.

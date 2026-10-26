# Spark X2.6 with packed quantized weights

This run removes the persistent FP16 prefill weight copy. GGUF matrices remain in their checkpoint encodings for both prefill and decode. FP8 safetensors now retain E4M3 bytes and BF16 channel scales. Quantized coefficients are decoded only in temporary compute tiles; no full persistent FP16 matrix is created. The removed prefill_weight_cache option is rejected.

Peak observed GPU memory: **9.50 GiB (9,724 MiB)**, compared with 16.63 GiB in the superseded cached-weight run. Resident weight allocations including rotary tables: 3.45 GiB. Persistent state: 4.66 GiB, almost entirely FP16 KV. The remaining memory covers workspaces, loaded kernels, graphs and CUDA overhead.

One request, prefixes 32 through 130944, 128 total output tokens, FP16 KV, one warmup and three retained samples per prefix. Loading and phase preparation are outside the timers. Native AKAS phase timing and llama.cpp server eval timing have different boundaries.

The comparison reuses sealed llama.cpp run `20261026101811969664-715fa9e0` from earlier on October 26, 2026. Its checkpoint bytes, prompt IDs, GPU, output count, KV dtype, warmup/repetition counts and pinned binary are unchanged. llama.cpp was not rerun for this implementation-only change.

AKAS decode medians exceed llama.cpp at 8/8 prefixes, and all retained decode samples exceed corresponding llama.cpp samples at 8/8. These results describe the measured workload, rather than guaranteeing all inputs or hardware.

| Prefix | AKAS prefill tok/s | llama.cpp prefill tok/s | AKAS decode tok/s | llama.cpp decode tok/s | Estimated prefill MFU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1,088.3 | 844.1 | 71.30 | 63.68 | 6.82% |
| 512 | 1,711.8 | 4,179.9 | 69.84 | 65.61 | 10.90% |
| 2,048 | 1,739.4 | 4,542.6 | 68.53 | 50.64 | 11.36% |
| 8,192 | 1,692.3 | 4,359.9 | 64.78 | 41.31 | 11.72% |
| 16,384 | 1,615.3 | 4,038.9 | 60.26 | 55.38 | 11.99% |
| 32,768 | 1,475.5 | 3,136.7 | 51.75 | 46.84 | 12.43% |
| 65,536 | 1,260.3 | 2,138.9 | 41.52 | 37.18 | 13.13% |
| 130,944 | 1,000.8 | 1,337.8 | 30.16 | 27.55 | 14.42% |

Prefill MFU estimates useful dense projections, one final-row vocabulary projection and causal QK/PV work, relative to [NVIDIA L4’s 121 TFLOP/s dense FP16 peak](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf). It excludes embedding lookups, elementwise work, padding, dequantization and cache/control traffic; it counts all layers even where intermediate chunks prune final-layer outputs. It is not measured hardware instruction utilization.

## Qualification

Storage tests verify original GGUF/FP8 bytes are preserved and no expanded quantized GPU weight matrices are allocated. Independent CUDA operation checks and BF16/FP8 small-model Torch references cover cache wrap, reorder, slot reuse, inactive requests and NaN padding. The actual 4B FP8 checkpoint also passes a finite-logit decode smoke check with native FP8 storage. The long-reduction FP8 projection matches the existing F16-operand/F32-accumulation GEMM bit for bit on all 81,920 tested outputs; its absolute qualification tolerance matches packed projections. Supplemental logs and generators are retained in the qualification run. Full-model cross-runtime greedy trajectory comparisons remain separate from timing in the canonical report.

## Reproduce

```sh
python -m tools.spark_benchmark compile --optimized --chunk 1024 --program build/spark/packed --out build/spark/packed-compile.json
python -m tools.spark_benchmark akas --chunk 1024 --program build/spark/packed --out build/spark/packed-run.json
python -m tools.spark_benchmark report --runs AKAS_RUN 20261026101811969664-715fa9e0 --out build/spark/packed-report
python -m tools.spark_mfu REPORT_DIRECTORY/report.json --checkpoint MODEL.gguf --out build/spark/mfu.json
```

The optimized option enables shared rotary tables and optimized attention. Quantized weights remain packed. Recompile older programs that enabled the removed FP16 prefill cache or used the former FP8-to-FP16 import layout.

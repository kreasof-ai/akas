# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar"></a>
## lfm2.5-24b-a2b-fp8 / akas / fp8 / cuda / float16 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json).

Peak device GiB: 51.1; peak host process RSS GiB: 5.5. Loading and in-process compilation are included in host RSS.

native LFM2 MoE requires CUDA and original weight layouts

Selected recipe: `{"chunk": 2048, "decode_prefill_token_budget": 256, "deferred": false, "graphs": true, "head": null, "id": "1bda28071fd5e5de", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {"attention": {"decode_precision": "float16", "minimum_split_groups": 256}, "convolution": {"fused_staging": true}, "linear": {"block_m": 32, "columns": 128, "floating_matmul": true, "minimum_rows": 1, "minimum_split_groups": 128, "packed_copy": true, "partition_maximum_rows": 4, "partitions": 8, "stage_inputs": true, "stages": 2, "threads": 256}, "moe": {"activation_dtype": "bfloat16", "block_m": 32, "fused_quantize_minimum_rows": 9, "fused_swiglu": true, "intermediate_dtype": "bfloat16", "max_block_m": 128, "packed": true, "packed_minimum_rows": 2048, "partition_maximum_rows": 1, "partitions": 2}, "normalization": {"fused_residual": true, "fused_staging": true}}, "prefill_requests": 16, "prefill_token_budget": 524288, "provider": "cuda", "target": "sm_90a", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | native-token-step-v2 | 64,458.8 | 319.3 | 298.0 | — | 298.0 | 31.8 | 3.1 | 1 | passed | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 87,152.9 | 932.5 | 801.5 | — | 801.5 | 94.0 | 4.3 | 1 | passed | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 92,201.5 | 2,207.9 | 1,605.4 | — | 1,605.4 | 355.4 | 7.2 | 1 | passed | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 90,179.5 | 6,343.1 | 2,995.4 | — | 2,995.4 | 1,453.5 | 10.1 | 1 | passed | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 89,432.5 | 14,799.1 | 4,065.9 | — | 4,065.9 | 5,862.4 | 17.3 | 1 | passed | [20261103091212903514-d753d574](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103091212903514-d753d574/run.json) |

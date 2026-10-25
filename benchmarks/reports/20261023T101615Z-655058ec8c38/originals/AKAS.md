# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-float16-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / float16 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json).

Peak device GiB: —; peak host process RSS GiB: 3.5. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "deferred": false, "graphs": true, "head": null, "id": "90b57aec951229c9", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {"attention": {"decoded_kv": false, "native_tail": true, "splits": 64}, "moe": {"fused_swiglu": true}}, "prefill_requests": 1, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 570.2 | 130.5 | 124.3 | 121.1 | 119.7 | 57.7 | 8.0 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 2,837.7 | 130.2 | 110.8 | 108.2 | 107.1 | 179.4 | 8.0 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 4,219.3 | 128.1 | 86.6 | 84.9 | 83.8 | 495.8 | 8.1 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,344.2 | 120.7 | 43.6 | 42.7 | 42.2 | 1,940.3 | 8.6 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,188.3 | 112.0 | 25.4 | 24.8 | 24.4 | 4,062.2 | 9.3 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,863.9 | 97.9 | 13.1 | 12.8 | 12.8 | 8,654.5 | 10.5 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,414.8 | 77.9 | 6.1 | 6.1 | 6.1 | 19,416.4 | 13.1 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,807.3 | 56.1 | 2.6 | 2.6 | 2.6 | 47,530.8 | 18.2 | 3 | passed | [20261023101615391936-42d4989e](../../runs/20261023101615391936-42d4989e/run.json) |

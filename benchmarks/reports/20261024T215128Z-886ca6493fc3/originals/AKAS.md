# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-2.6b-q4_0-webgpu-float16-ar"></a>
## lfm2.5-2.6b / akas / q4_0 / webgpu / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261024215128041404-5569d5c7](../../runs/20261024215128041404-5569d5c7/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {"attention": {"adaptive_splits": true, "splits": 64}, "linear": {"stage_weights": true}}, "prefill_requests": 1, "trials": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 333.4 | 166.9 | 149.4 | — | — | — | — | 3 | passed | [20261024215128041404-5569d5c7](../../runs/20261024215128041404-5569d5c7/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 820.8 | 165.0 | 91.9 | — | — | — | — | 3 | passed | [20261024215128041404-5569d5c7](../../runs/20261024215128041404-5569d5c7/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 794.0 | 158.6 | 37.9 | — | — | — | — | 3 | passed | [20261024215128041404-5569d5c7](../../runs/20261024215128041404-5569d5c7/run.json) |

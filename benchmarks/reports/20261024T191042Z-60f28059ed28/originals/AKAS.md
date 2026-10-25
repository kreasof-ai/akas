# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-2.6b-q4_0-webgpu-float16-ar"></a>
## lfm2.5-2.6b / akas / q4_0 / webgpu / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "model_options": {"activation_quantization": "mixed", "kv_dtype": "float16", "optimized": true}, "policy": {"attention": {"adaptive_splits": true, "splits": 128}, "linear": {"stage_inputs": true, "stage_weights": true}}, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 458.0 | 169.8 | 156.5 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 974.8 | 168.5 | 100.1 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 953.1 | 163.3 | 43.7 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 835.6 | 152.9 | 12.0 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 712.8 | 139.8 | 5.4 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 556.0 | 122.2 | 2.1 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 386.2 | 97.1 | 0.7 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 239.5 | 68.2 | 0.2 | — | — | — | — | 1 | passed | [20261025121928791585-816cf540](../../runs/20261025121928791585-816cf540/run.json) |

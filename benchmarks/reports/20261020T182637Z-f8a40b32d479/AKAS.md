# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json).

Peak device GiB: —; peak host process RSS GiB: 4.1. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "deferred": false, "graphs": true, "head": null, "id": "c4901473e55177a4", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {"attention": {"decoded_kv": true, "splits": 64}, "moe": {"fused_swiglu": true}}, "prefill_requests": 1, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 539.0 | 130.2 | 123.7 | 120.4 | 119.6 | 59.9 | 8.0 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 989.8 | 130.3 | 85.8 | 84.4 | 83.9 | 512.5 | 8.0 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,875.7 | 128.4 | 75.3 | 74.1 | 73.8 | 708.5 | 8.1 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,145.5 | 122.8 | 42.5 | 41.8 | 41.7 | 1,996.3 | 8.5 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 3,957.2 | 118.2 | 24.5 | 24.1 | 24.2 | 4,187.6 | 8.8 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,616.9 | 109.3 | 12.5 | 12.3 | 12.4 | 9,159.3 | 9.5 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,093.0 | 94.8 | 5.7 | 5.6 | 5.6 | 21,308.2 | 10.9 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,426.9 | 75.3 | 2.3 | 2.3 | 2.3 | 54,444.2 | 13.6 | 3 | passed | [20261020182637613063-8a87f1a5](../../runs/20261020182637613063-8a87f1a5/run.json) |

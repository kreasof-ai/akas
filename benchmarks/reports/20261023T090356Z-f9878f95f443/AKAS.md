# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-float16-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / float16 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json).

Peak device GiB: —; peak host process RSS GiB: 3.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "deferred": false, "graphs": true, "head": null, "id": "8b123f530c46906c", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {"attention": {"decoded_kv": false, "splits": 64}, "moe": {"fused_swiglu": true}}, "prefill_requests": 1, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 526.3 | 130.3 | 123.6 | 120.5 | 119.9 | 60.9 | 7.9 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 2,870.2 | 129.8 | 110.7 | 108.0 | 107.5 | 179.6 | 8.0 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 4,159.1 | 127.8 | 86.1 | 84.4 | 84.1 | 492.6 | 8.1 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,259.0 | 120.3 | 43.0 | 42.1 | 42.1 | 1,950.2 | 8.6 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,050.2 | 111.6 | 24.7 | 24.1 | 24.2 | 4,108.8 | 9.2 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,671.1 | 97.8 | 12.5 | 12.4 | 12.3 | 9,061.6 | 10.5 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,159.3 | 77.9 | 5.7 | 5.6 | 5.7 | 20,867.5 | 13.1 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,456.1 | 55.9 | 2.3 | 2.3 | 2.3 | 53,837.2 | 18.2 | 3 | passed | [20261023072001816909-fb5b388f](../../runs/20261023072001816909-fb5b388f/run.json) |

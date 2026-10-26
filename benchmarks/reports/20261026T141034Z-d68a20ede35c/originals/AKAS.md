# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="spark-x2.6-4b-q4_k_m-cuda-float16-ar"></a>
## spark-x2.6-4b / akas / q4_k_m / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 2048, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,085.6 | 71.2 | 70.6 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 4,238.4 | 69.8 | 66.0 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 4,632.6 | 68.5 | 55.7 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,406.6 | 64.9 | 33.5 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,061.0 | 60.3 | 20.9 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,507.1 | 51.9 | 10.9 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 2,724.4 | 41.6 | 4.7 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 1,879.4 | 30.1 | 1.7 | — | — | — | — | 3 | passed | [20261026154130235361-4e09b6ff](../../runs/20261026154130235361-4e09b6ff/run.json) |

# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="spark-x2.6-4b-q4_k_m-cuda-float16-ar"></a>
## spark-x2.6-4b / akas / q4_k_m / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 256, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 677.6 | 65.3 | 64.3 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 1,545.8 | 63.3 | 54.8 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,523.5 | 61.6 | 37.6 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 1,448.1 | 57.4 | 16.3 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 1,350.6 | 51.2 | 8.8 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 1,190.7 | 43.1 | 4.2 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 964.8 | 32.9 | 1.8 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 714.0 | 22.7 | 0.7 | — | — | — | — | 3 | passed | [20261025235635419062-730dd488](../../runs/20261025235635419062-730dd488/run.json) |

# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-spark-x2.6-4b-llama-cuda-q4_k_m-ar"></a>
## spark-x2.6-4b / llama / q4_k_m / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 839.9 | 63.8 | — | — | 63.1 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |
| 1 | 512 | 128 | llama-server-eval-v1 | 4,195.7 | 65.7 | — | — | 62.3 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 4,536.1 | 50.7 | — | — | 43.3 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |
| 1 | 8192 | 128 | llama-server-eval-v1 | 4,351.3 | 41.0 | — | — | 25.7 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |
| 1 | 16384 | 128 | llama-server-eval-v1 | 4,041.3 | 55.4 | — | — | 20.1 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |
| 1 | 32768 | 128 | llama-server-eval-v1 | 3,135.5 | 46.9 | — | — | 9.7 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |
| 1 | 65536 | 128 | llama-server-eval-v1 | 2,139.1 | 37.3 | — | — | 3.8 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |
| 1 | 130944 | 128 | llama-server-eval-v1 | 1,336.2 | 27.6 | — | — | 1.2 | — | — | 3 | passed | [20261026141034121130-479a28cb](../../runs/20261026141034121130-479a28cb/run.json) |

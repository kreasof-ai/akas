# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm3-1.7b-llama-vulkan-q4_0-ar"></a>
## lfm3-1.7b / llama / q4_0 / vulkan / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261028103203103736-098e0675](../../runs/20261028103203103736-098e0675/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 1,184.7 | 218.9 | — | — | 210.2 | — | — | 3 | passed | [20261028103203103736-098e0675](../../runs/20261028103203103736-098e0675/run.json) |
| 1 | 512 | 128 | llama-server-eval-v1 | 3,693.0 | 214.4 | — | — | 174.6 | — | — | 3 | passed | [20261028103203103736-098e0675](../../runs/20261028103203103736-098e0675/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 2,435.7 | 209.6 | — | — | 88.3 | — | — | 3 | passed | [20261028103203103736-098e0675](../../runs/20261028103203103736-098e0675/run.json) |
| 1 | 8064 | 128 | llama-server-eval-v1 | 1,007.3 | 196.4 | — | — | 14.8 | — | — | 3 | passed | [20261028103203103736-098e0675](../../runs/20261028103203103736-098e0675/run.json) |

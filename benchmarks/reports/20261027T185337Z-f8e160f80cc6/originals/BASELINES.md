# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm3-1.7b-llama-cuda-q4_0-ar"></a>
## lfm3-1.7b / llama / q4_0 / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 3,206.4 | 190.1 | — | — | 188.3 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |
| 1 | 512 | 128 | llama-server-eval-v1 | 13,559.7 | 186.1 | — | — | 177.4 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 12,957.2 | 183.5 | — | — | 150.1 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |
| 1 | 8192 | 128 | llama-server-eval-v1 | 12,381.4 | 172.8 | — | — | 91.3 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |
| 1 | 16384 | 128 | llama-server-eval-v1 | 11,565.9 | 160.3 | — | — | 57.7 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |
| 1 | 32768 | 128 | llama-server-eval-v1 | 9,988.2 | 140.2 | — | — | 30.4 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |
| 1 | 65536 | 128 | llama-server-eval-v1 | 6,517.1 | 110.0 | — | — | 11.4 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |
| 1 | 130944 | 128 | llama-server-eval-v1 | 3,942.3 | 79.3 | — | — | 3.7 | — | — | 3 | passed | [20261027185932613950-5409e768](../../runs/20261027185932613950-5409e768/run.json) |

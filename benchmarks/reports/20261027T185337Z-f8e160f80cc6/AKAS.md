# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm3-1.7b-q4_0-cuda-float16-ar"></a>
## lfm3-1.7b / akas / q4_0 / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 2,553.1 | 189.4 | 187.2 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 9,528.4 | 188.6 | 176.0 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 9,810.6 | 186.0 | 143.6 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 9,418.7 | 176.4 | 80.5 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 8,616.0 | 165.2 | 47.9 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 7,370.3 | 142.1 | 24.0 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 5,687.5 | 113.8 | 10.1 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 3,879.9 | 82.4 | 3.6 | — | — | — | — | 3 | passed | [20261027185337181202-06e9f76f](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261027185337181202-06e9f76f/run.json) |

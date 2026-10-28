# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm3-1.7b-q4_0-cuda-float16-ar"></a>
## lfm3-1.7b / akas / q4_0 / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 1024, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,315.6 | 204.6 | 203.0 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 13,932.3 | 203.7 | 194.1 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 14,363.0 | 200.9 | 165.1 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 13,293.5 | 190.0 | 99.6 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 12,041.7 | 177.4 | 61.6 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 10,176.7 | 150.9 | 31.5 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 7,655.9 | 119.5 | 13.3 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 5,081.0 | 85.4 | 4.7 | — | — | — | — | 3 | passed | [20261028072145009000-65166f09](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028072145009000-65166f09/run.json) |

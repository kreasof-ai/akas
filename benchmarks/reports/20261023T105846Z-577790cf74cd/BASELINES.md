# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar"></a>
## lfm2.5-24b-a2b / llama / q4_0 / cuda / default / ar

Profile status: **not_measured**; qualification: **not_measured**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

No matching full-workload baseline before campaign start

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 8192 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 130944 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar"></a>
## lfm2.5-24b-a2b / llama / q4_0 / vulkan / default / ar

Profile status: **not_measured**; qualification: **not_measured**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

No matching full-workload baseline before campaign start

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 8192 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 130944 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

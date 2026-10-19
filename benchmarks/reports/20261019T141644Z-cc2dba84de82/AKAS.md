# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; the measured LFM floating path executes FP16. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-default-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / default / ar

Profile status: **pending**; qualification: **not_measured**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

No qualified deployment retained

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 512 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 8192 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 16384 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 32768 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 65536 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 130944 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-24b-a2b-q4_0-webgpu-default-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / webgpu / default / ar

Profile status: **pending**; qualification: **not_measured**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

No qualified deployment retained

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 512 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 8192 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 16384 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 32768 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 65536 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 130944 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar"></a>
## lfm2.5-24b-a2b / llama / q4_0 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json).

Peak device GiB: 15.6; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 764.8 | 135.0 | — | — | 130.1 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |
| 1 | 512 | 128 | llama-server-eval-v1 | 3,572.6 | 133.6 | — | — | 116.8 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 3,510.0 | 131.2 | — | — | 82.4 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |
| 1 | 8192 | 128 | llama-server-eval-v1 | 3,455.1 | 121.7 | — | — | 37.4 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |
| 1 | 16384 | 128 | llama-server-eval-v1 | 3,323.3 | 111.1 | — | — | 21.0 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |
| 1 | 32768 | 128 | llama-server-eval-v1 | 3,080.1 | 95.3 | — | — | 10.7 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |
| 1 | 65536 | 128 | llama-server-eval-v1 | 2,591.0 | 74.2 | — | — | 4.7 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |
| 1 | 130944 | 128 | llama-server-eval-v1 | 1,623.0 | 51.2 | — | — | 1.5 | — | — | 3 | passed | [20261023080534992856-19529564](../../runs/20261023080534992856-19529564/run.json) |

<a id="baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar"></a>
## lfm2.5-24b-a2b / llama / q4_0 / vulkan / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json).

Peak device GiB: 15.4; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 431.6 | 116.9 | — | — | 110.3 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |
| 1 | 512 | 128 | llama-server-eval-v1 | 2,713.6 | 116.3 | — | — | 99.8 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 2,695.9 | 114.6 | — | — | 68.4 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |
| 1 | 8192 | 128 | llama-server-eval-v1 | 2,571.0 | 108.4 | — | — | 29.3 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |
| 1 | 16384 | 128 | llama-server-eval-v1 | 2,378.3 | 100.7 | — | — | 15.7 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |
| 1 | 32768 | 128 | llama-server-eval-v1 | 2,095.0 | 88.2 | — | — | 7.5 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |
| 1 | 65536 | 128 | llama-server-eval-v1 | 1,678.5 | 71.0 | — | — | 3.1 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |
| 1 | 130944 | 128 | llama-server-eval-v1 | 1,243.2 | 51.7 | — | — | 1.2 | — | — | 3 | passed | [20261023084715832350-d710881c](../../runs/20261023084715832350-d710881c/run.json) |

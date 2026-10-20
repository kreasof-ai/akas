# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; the measured LFM floating path executes FP16. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar"></a>
## lfm2.5-24b-a2b / llama / q4_0 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json).

Peak device GiB: 15.6; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 763.0 | 133.2 | — | — | 128.4 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| 1 | 512 | 128 | llama-server-eval-v1 | 3,619.0 | 131.9 | — | — | 115.7 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 3,530.5 | 129.3 | — | — | 81.9 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| 1 | 8192 | 128 | llama-server-eval-v1 | 3,490.7 | 120.2 | — | — | 37.6 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| 1 | 16384 | 128 | llama-server-eval-v1 | 3,339.3 | 109.5 | — | — | 21.1 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| 1 | 32768 | 128 | llama-server-eval-v1 | 3,076.8 | 93.8 | — | — | 10.6 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| 1 | 65536 | 128 | llama-server-eval-v1 | 2,576.6 | 72.9 | — | — | 4.7 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |
| 1 | 130944 | 128 | llama-server-eval-v1 | 1,624.9 | 50.6 | — | — | 1.5 | — | — | 3 | passed | [20261018202355478915-6adad97d](../../runs/20261018202355478915-6adad97d/run.json) |

<a id="baseline-lfm2.5-24b-a2b-llama-vulkan-q4_0-ar"></a>
## lfm2.5-24b-a2b / llama / q4_0 / vulkan / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json).

Peak device GiB: 15.4; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 425.4 | 115.6 | — | — | 108.9 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |
| 1 | 512 | 128 | llama-server-eval-v1 | 2,706.4 | 114.0 | — | — | 98.1 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 2,702.6 | 112.9 | — | — | 67.9 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |
| 1 | 8192 | 128 | llama-server-eval-v1 | 2,578.1 | 106.4 | — | — | 29.2 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |
| 1 | 16384 | 128 | llama-server-eval-v1 | 2,393.8 | 98.0 | — | — | 15.7 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |
| 1 | 32768 | 128 | llama-server-eval-v1 | 2,118.5 | 86.5 | — | — | 7.5 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |
| 1 | 65536 | 128 | llama-server-eval-v1 | 1,712.5 | 69.8 | — | — | 3.2 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |
| 1 | 130944 | 128 | llama-server-eval-v1 | 1,251.9 | 51.0 | — | — | 1.2 | — | — | 3 | passed | [20261018203703691867-28072a0a](../../runs/20261018203703691867-28072a0a/run.json) |

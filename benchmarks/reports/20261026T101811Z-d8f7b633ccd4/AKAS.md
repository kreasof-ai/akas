# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="spark-x2.6-4b-q4_k_m-cuda-float16-ar"></a>
## spark-x2.6-4b / akas / q4_k_m / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 1024, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,088.3 | 71.3 | 70.7 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 1,711.8 | 69.8 | 60.5 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,739.4 | 68.5 | 42.2 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 1,692.3 | 64.8 | 18.8 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 1,615.3 | 60.3 | 10.4 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 1,475.5 | 51.7 | 5.2 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 1,260.3 | 41.5 | 2.3 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 1,000.8 | 30.2 | 0.9 | — | — | — | — | 3 | passed | [20261026111637611638-9ff71082](https://github.com/kreasof-ai/akas/blob/c20246b485e2943b1afb42a3bb79c02a30595627/benchmarks/runs/20261026111637611638-9ff71082/run.json) |

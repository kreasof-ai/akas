# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-ar"></a>
## lfm2.5-24b-a2b / llama / q4_0 / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json).

Peak device GiB: —; peak host process RSS GiB: 10.7. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-real-text-v1 | 683.1 | 130.0 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |
| 1 | 512 | 128 | llama-server-eval-real-text-v1 | 2,198.6 | 132.6 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |
| 1 | 2048 | 128 | llama-server-eval-real-text-v1 | 3,279.2 | 129.9 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |
| 1 | 8192 | 128 | llama-server-eval-real-text-v1 | 3,208.3 | 121.1 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |
| 1 | 16384 | 128 | llama-server-eval-real-text-v1 | 3,112.6 | 111.0 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |
| 1 | 32768 | 128 | llama-server-eval-real-text-v1 | 2,897.3 | 95.1 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |
| 1 | 65536 | 128 | llama-server-eval-real-text-v1 | 2,495.2 | 74.2 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |
| 1 | 130944 | 128 | llama-server-eval-real-text-v1 | 1,593.4 | 51.3 | — | — | — | — | — | 1 | passed | [20261023162435892403-5cca5399](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162435892403-5cca5399/run.json) |

<a id="baseline-lfm2.5-24b-a2b-llama-cuda-q4_0-dspark"></a>
## lfm2.5-24b-a2b / llama / q4_0 / cuda / float16 / dspark

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json).

Peak device GiB: —; peak host process RSS GiB: 10.7. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-real-text-v1 | 650.2 | 80.3 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |
| 1 | 512 | 128 | llama-server-eval-real-text-v1 | 1,705.0 | 85.8 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |
| 1 | 2048 | 128 | llama-server-eval-real-text-v1 | 2,544.8 | 171.2 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |
| 1 | 8192 | 128 | llama-server-eval-real-text-v1 | 2,858.9 | 131.8 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |
| 1 | 16384 | 128 | llama-server-eval-real-text-v1 | 2,912.5 | 141.1 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |
| 1 | 32768 | 128 | llama-server-eval-real-text-v1 | 2,714.9 | 64.6 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |
| 1 | 65536 | 128 | llama-server-eval-real-text-v1 | 2,315.3 | 41.4 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |
| 1 | 130944 | 128 | llama-server-eval-real-text-v1 | 1,508.1 | 28.1 | — | — | — | — | — | 1 | passed | [20261023162947335687-fbeb0b96](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023162947335687-fbeb0b96/run.json) |

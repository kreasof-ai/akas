# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-float16-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json).

Peak device GiB: —; peak host process RSS GiB: 0.9. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "confidence_threshold": 0.5, "draft_weight_dtype": "q8_0", "graphs": true, "kv_dtype": "float16", "mode": "ar", "prefill_requests": 1, "trials": 0}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 581.2 | 131.8 | 125.7 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 1,090.3 | 131.2 | 89.0 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,631.2 | 128.9 | 72.6 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 3,407.8 | 121.4 | 37.1 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,259.0 | 112.7 | 25.7 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,956.6 | 98.8 | 13.4 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,465.2 | 78.3 | 6.2 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,771.7 | 55.7 | 2.6 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |

<a id="lfm2.5-24b-a2b-q4_0-cuda-float16-dspark"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / float16 / dspark

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json).

Peak device GiB: —; peak host process RSS GiB: 0.9. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "confidence_threshold": 0.5, "draft_weight_dtype": "q8_0", "graphs": true, "kv_dtype": "float16", "mode": "dspark", "prefill_requests": 1, "trials": 0}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 500.4 | 64.9 | 63.4 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 2,095.4 | 202.9 | 147.1 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 3,430.9 | 256.1 | 117.1 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,150.4 | 123.5 | 42.6 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,092.0 | 95.5 | 24.0 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,804.9 | 54.9 | 11.7 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,330.2 | 29.7 | 5.3 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,715.7 | 16.8 | 2.3 | — | — | — | — | 1 | passed | [20261023155400228961-efa6319e](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023155400228961-efa6319e/run.json) |

<a id="lfm2.5-24b-a2b-q4_0-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / fp8 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json).

Peak device GiB: —; peak host process RSS GiB: 0.9. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "confidence_threshold": 0.5, "draft_weight_dtype": "q8_0", "graphs": true, "kv_dtype": "fp8", "mode": "ar", "prefill_requests": 1, "trials": 0}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 588.2 | 131.2 | 125.2 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 1,106.2 | 130.7 | 89.2 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,648.7 | 129.0 | 72.8 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 3,406.8 | 123.2 | 37.3 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,220.0 | 118.6 | 25.8 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,934.4 | 109.8 | 13.5 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,445.7 | 95.0 | 6.3 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,755.7 | 75.1 | 2.6 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |

<a id="lfm2.5-24b-a2b-q4_0-cuda-fp8-dspark"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / fp8 / dspark

Profile status: **passed**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json).

Peak device GiB: —; peak host process RSS GiB: 0.9. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "confidence_threshold": 0.5, "draft_weight_dtype": "q8_0", "graphs": true, "kv_dtype": "fp8", "mode": "dspark", "prefill_requests": 1, "trials": 0}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 510.1 | 66.6 | 64.9 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 2,110.1 | 196.4 | 144.0 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 3,409.4 | 191.0 | 101.1 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,132.8 | 101.7 | 39.6 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,052.2 | 108.7 | 24.6 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,770.4 | 53.2 | 11.6 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,309.4 | 29.4 | 5.3 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,680.5 | 11.0 | 2.1 | — | — | — | — | 1 | passed | [20261023161702005532-82eb67d4](https://github.com/akbar2habibullah/akas/blob/dccbca5fbffe8e37fbafc9ef7181b2bf5e97f416/benchmarks/runs/20261023161702005532-82eb67d4/run.json) |

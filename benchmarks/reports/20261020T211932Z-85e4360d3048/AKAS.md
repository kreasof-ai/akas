# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json).

Peak device GiB: —; peak host process RSS GiB: 4.1. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 4096, "deferred": false, "graphs": true, "head": null, "id": "c4901473e55177a4", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {"attention": {"decoded_kv": true, "splits": 64}, "moe": {"fused_swiglu": true}}, "prefill_requests": 1, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 537.5 | 130.2 | 123.6 | 120.2 | 119.8 | 59.5 | 7.9 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 2,739.6 | 130.1 | 109.9 | 107.0 | 107.2 | 184.6 | 8.0 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 4,157.8 | 128.3 | 86.3 | 84.5 | 85.0 | 486.3 | 8.0 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 4,311.3 | 122.4 | 43.6 | 42.7 | 43.3 | 1,889.8 | 8.4 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 4,142.0 | 117.3 | 25.4 | 24.8 | 25.3 | 3,945.3 | 8.8 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 3,847.4 | 108.6 | 13.2 | 13.0 | 13.1 | 8,532.4 | 9.5 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 3,424.6 | 94.8 | 6.3 | 6.2 | 6.2 | 19,304.1 | 10.8 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 2,790.2 | 75.3 | 2.6 | 2.6 | 2.6 | 46,831.1 | 13.5 | 3 | passed | [20261020211932758011-47b0b378](https://github.com/akbar2habibullah/akas/blob/00ffeef39d625c6fb44146088afd9b3ffe968b27/benchmarks/runs/20261020211932758011-47b0b378/run.json) |

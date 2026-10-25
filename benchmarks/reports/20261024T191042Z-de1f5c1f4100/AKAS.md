# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-2.6b-q4_0-webgpu-float16-ar"></a>
## lfm2.5-2.6b / akas / q4_0 / webgpu / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {"attention": {"adaptive_splits": true, "splits": 64}, "linear": {"stage_weights": true}}, "prefill_requests": 1, "trials": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 323.2 | 166.1 | 148.2 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 816.3 | 164.8 | 91.6 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 792.8 | 158.9 | 37.8 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 693.9 | 143.4 | 10.1 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 594.3 | 124.5 | 4.5 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 465.3 | 98.1 | 1.8 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 287.9 | 68.7 | 0.6 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 168.5 | 43.1 | 0.2 | — | — | — | — | 1 | passed | [20261024215325218226-fd38f777](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024215325218226-fd38f777/run.json) |

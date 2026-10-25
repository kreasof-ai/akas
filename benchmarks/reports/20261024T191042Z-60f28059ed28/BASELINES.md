# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm2.5-2.6b-llama-vulkan-q4_0-ar"></a>
## lfm2.5-2.6b / llama / q4_0 / vulkan / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 512, "graphs": true, "prefill_requests": 1, "trials": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-real-text-v1 | 921.0 | 165.4 | — | — | 159.2 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |
| 1 | 512 | 128 | llama-server-eval-real-text-v1 | 2,596.5 | 161.5 | — | — | 129.9 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |
| 1 | 2048 | 128 | llama-server-eval-real-text-v1 | 2,470.2 | 160.8 | — | — | 78.9 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |
| 1 | 8192 | 128 | llama-server-eval-real-text-v1 | 1,997.9 | 151.5 | — | — | 25.9 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |
| 1 | 16384 | 128 | llama-server-eval-real-text-v1 | 1,556.3 | 144.0 | — | — | 11.2 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |
| 1 | 32768 | 128 | llama-server-eval-real-text-v1 | 1,068.5 | 127.9 | — | — | 4.0 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |
| 1 | 65536 | 128 | llama-server-eval-real-text-v1 | 655.5 | 105.2 | — | — | 1.3 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |
| 1 | 130944 | 128 | llama-server-eval-real-text-v1 | 367.6 | 78.2 | — | — | 0.4 | — | — | 1 | passed | [20261024191042883833-974e6fff](https://github.com/akbar2habibullah/akas/blob/4183d2d34392cd6a928acf2f717e6ca672e01402/benchmarks/runs/20261024191042883833-974e6fff/run.json) |

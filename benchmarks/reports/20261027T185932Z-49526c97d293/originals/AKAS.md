# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm3-1.7b-q4_0-cuda-float16-ar"></a>
## lfm3-1.7b / akas / q4_0 / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 1024, "graphs": true, "prefill_requests": 1, "trials": 1}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,266.6 | 200.8 | 199.2 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 12,002.3 | 199.6 | 188.6 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 12,924.1 | 196.4 | 159.0 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 11,990.7 | 186.4 | 93.8 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 10,939.3 | 174.1 | 57.5 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 9,299.1 | 148.6 | 29.2 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 7,128.9 | 117.9 | 12.5 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 4,821.3 | 84.5 | 4.5 | — | — | — | — | 3 | passed | [20261027211545023428-59d02ce9](../../runs/20261027211545023428-59d02ce9/run.json) |

# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; the measured LFM floating path executes FP16. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-230m-bf16-cuda-default-ar"></a>
## lfm2.5-230m / akas / bf16 / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json).

Peak device GiB: 7.5; peak host process RSS GiB: 1.7. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 8,918.2 | 440.8 | 438.8 | 438.8 | 171.6 | 277.4 | 3.7 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 8,612.2 | 404.3 | 231.9 | 231.9 | 231.9 | 237.8 | 2.5 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 453.9 | 1,104.9 | 690.2 | 690.2 | 690.2 | 282.0 | 3.6 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,876.3 | 1,306.8 | 358.3 | 358.3 | 358.3 | 1,040.1 | 3.1 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 35,601.2 | 2,899.3 | 2,863.4 | 2,863.4 | 2,863.4 | 14.4 | 5.5 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 29,971.5 | 2,246.7 | 1,025.2 | 1,025.2 | 1,025.2 | 1,093.3 | 7.1 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 8,927.5 | 12,856.6 | 9,507.8 | 9,507.8 | 9,507.8 | 229.4 | 6.6 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 56,574.6 | 3,898.0 | 1,861.0 | 1,861.0 | 1,861.0 | 2,316.8 | 16.6 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 23,772.3 | 22,946.6 | 18,602.8 | 18,602.8 | 18,602.8 | 344.6 | 13.1 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 55,331.5 | 4,336.5 | 1,930.6 | 1,930.6 | 1,930.6 | 9,475.4 | 60.3 | 1 | passed | [20261018130852567320-14df8906](../../runs/20261018130852567320-14df8906/run.json) |

<a id="lfm2.5-230m-bf16-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / bf16 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json).

Peak device GiB: 4.3; peak host process RSS GiB: 1.7. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 9,037.3 | 439.1 | 437.2 | 437.2 | 164.9 | 300.7 | 3.7 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 9,135.0 | 404.0 | 237.7 | 237.7 | 237.7 | 224.2 | 2.5 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 450.0 | 1,107.5 | 689.0 | 689.0 | 689.0 | 284.4 | 3.6 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,751.6 | 1,311.4 | 354.5 | 354.5 | 354.5 | 1,056.8 | 3.1 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 35,578.9 | 2,892.0 | 2,856.2 | 2,856.2 | 2,856.2 | 14.4 | 5.5 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 29,987.2 | 2,283.3 | 1,033.0 | 1,033.0 | 1,033.0 | 1,092.7 | 7.0 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 10,058.0 | 12,792.6 | 9,764.2 | 9,764.2 | 9,764.2 | 203.6 | 6.4 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 57,379.1 | 4,037.8 | 1,906.3 | 1,906.3 | 1,906.3 | 2,284.3 | 16.0 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 23,624.3 | 23,066.2 | 18,657.7 | 18,657.7 | 18,657.7 | 346.8 | 13.1 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 56,021.7 | 4,626.7 | 1,999.8 | 1,999.8 | 1,999.8 | 9,358.7 | 56.6 | 1 | passed | [20261018131006740252-43da420a](../../runs/20261018131006740252-43da420a/run.json) |

<a id="lfm2.5-230m-bf16-webgpu-default-ar"></a>
## lfm2.5-230m / akas / bf16 / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json).

Peak device GiB: 14.0; peak host process RSS GiB: 2.0. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "a60d051026a6a775", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,032.4 | 283.3 | 279.0 | 279.0 | 128.5 | 458.3 | 4.2 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,124.8 | 248.6 | 86.8 | 86.8 | 86.8 | 963.9 | 4.0 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 423.9 | 814.1 | 553.0 | 553.0 | 553.0 | 301.9 | 4.9 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 771.0 | 701.4 | 45.1 | 45.1 | 45.1 | 10,625.8 | 5.7 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,320.9 | 1,544.6 | 1,393.4 | 1,393.4 | 1,393.4 | 154.2 | 10.4 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 2,524.5 | 1,029.0 | 136.9 | 136.9 | 136.9 | 12,980.0 | 15.5 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,693.2 | 2,070.5 | 1,828.5 | 1,828.5 | 1,828.5 | 554.5 | 33.7 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 4,287.5 | 3,401.2 | 2,856.9 | 2,856.9 | 2,856.9 | 1,910.7 | 85.3 | 1 | passed | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018131122206190-b71485b2](../../runs/20261018131122206190-b71485b2/run.json) |

<a id="lfm2.5-230m-bf16-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / bf16 / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json).

Peak device GiB: 4.4; peak host process RSS GiB: 2.0. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "74d933f0b09e199a", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,003.6 | 281.6 | 277.2 | 277.2 | 151.9 | 280.5 | 4.4 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,895.4 | 230.6 | 78.5 | 78.5 | 78.5 | 1,080.5 | 4.3 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 424.2 | 811.6 | 551.9 | 551.9 | 551.9 | 301.8 | 4.9 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 741.5 | 647.3 | 43.3 | 43.3 | 43.3 | 11,048.2 | 6.2 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,257.5 | 1,521.7 | 1,372.2 | 1,372.2 | 1,372.2 | 157.2 | 10.5 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 2,255.5 | 918.3 | 122.3 | 122.3 | 122.3 | 14,528.2 | 17.4 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,681.3 | 2,030.6 | 1,796.8 | 1,796.8 | 1,796.8 | 556.3 | 33.7 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 4,248.3 | 3,328.3 | 2,801.5 | 2,801.5 | 2,801.5 | 1,928.3 | 87.0 | 1 | passed | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018131330864843-039fb80c](../../runs/20261018131330864843-039fb80c/run.json) |

<a id="lfm2.5-230m-f16-cuda-default-ar"></a>
## lfm2.5-230m / akas / f16 / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json).

Peak device GiB: 7.5; peak host process RSS GiB: 2.0. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 8,362.9 | 443.9 | 441.5 | 441.5 | 182.3 | 272.7 | 3.4 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 8,104.4 | 407.2 | 226.7 | 226.7 | 226.7 | 252.7 | 2.5 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 528.0 | 818.0 | 593.0 | 593.0 | 593.0 | 242.4 | 4.9 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,641.6 | 883.5 | 310.9 | 310.9 | 310.9 | 1,072.0 | 4.5 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 38,653.2 | 3,089.1 | 3,052.0 | 3,052.0 | 3,052.0 | 13.2 | 5.2 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 29,097.1 | 2,340.8 | 1,027.0 | 1,027.0 | 1,027.0 | 1,126.2 | 6.8 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 8,222.2 | 12,863.4 | 9,299.0 | 9,299.0 | 9,299.0 | 249.1 | 6.8 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 54,864.5 | 3,774.3 | 1,803.4 | 1,803.4 | 1,803.4 | 2,389.0 | 17.1 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 26,430.9 | 22,873.7 | 18,926.7 | 18,926.7 | 18,926.7 | 309.9 | 12.8 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 53,532.9 | 4,026.3 | 1,833.8 | 1,833.8 | 1,833.8 | 9,793.8 | 64.9 | 1 | passed | [20261018131544975070-4cbc5243](../../runs/20261018131544975070-4cbc5243/run.json) |

<a id="lfm2.5-230m-f16-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / f16 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json).

Peak device GiB: 4.3; peak host process RSS GiB: 2.0. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 8,417.8 | 444.4 | 442.1 | 442.1 | 174.8 | 286.1 | 3.5 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 8,583.2 | 406.6 | 232.3 | 232.3 | 232.3 | 238.6 | 2.5 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 383.6 | 818.7 | 536.6 | 536.6 | 536.6 | 333.7 | 4.9 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,525.7 | 888.5 | 308.4 | 308.4 | 308.4 | 1,088.5 | 4.5 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 38,229.5 | 3,103.6 | 3,065.3 | 3,065.3 | 3,065.3 | 13.4 | 5.2 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 29,097.0 | 2,390.4 | 1,036.3 | 1,036.3 | 1,036.3 | 1,126.2 | 6.7 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 12,369.5 | 12,909.5 | 10,302.1 | 10,302.1 | 10,302.1 | 165.6 | 6.1 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 55,328.3 | 3,970.9 | 1,855.1 | 1,855.1 | 1,855.1 | 2,369.0 | 16.3 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 25,142.2 | 23,139.3 | 18,931.4 | 18,931.4 | 18,931.4 | 325.8 | 12.9 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 53,869.2 | 4,438.1 | 1,921.0 | 1,921.0 | 1,921.0 | 9,732.6 | 59.0 | 1 | passed | [20261018131656581346-a1667c54](../../runs/20261018131656581346-a1667c54/run.json) |

<a id="lfm2.5-230m-f16-webgpu-default-ar"></a>
## lfm2.5-230m / akas / f16 / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json).

Peak device GiB: 14.0; peak host process RSS GiB: 2.3. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "a60d051026a6a775", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,932.7 | 261.1 | 254.5 | 254.5 | 115.0 | 550.0 | 4.4 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,725.5 | 242.1 | 74.8 | 74.8 | 74.8 | 1,186.9 | 4.1 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 448.3 | 229.7 | 205.0 | 205.0 | 205.0 | 285.5 | 17.4 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 758.4 | 225.3 | 39.2 | 39.2 | 39.2 | 10,801.8 | 17.8 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,352.9 | 874.3 | 826.8 | 826.8 | 826.8 | 152.7 | 18.3 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 2,477.8 | 720.5 | 127.6 | 127.6 | 127.6 | 13,224.8 | 22.2 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 4,636.4 | 2,450.0 | 2,179.1 | 2,179.1 | 2,179.1 | 441.7 | 28.1 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 4,327.4 | 3,344.5 | 2,821.4 | 2,821.4 | 2,821.4 | 1,893.0 | 86.4 | 1 | passed | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018131810905834-8f8a2aba](../../runs/20261018131810905834-8f8a2aba/run.json) |

<a id="lfm2.5-230m-f16-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / f16 / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json).

Peak device GiB: 4.4; peak host process RSS GiB: 2.5. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "74d933f0b09e199a", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,922.7 | 258.4 | 251.9 | 251.9 | 153.5 | 265.5 | 4.5 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,566.8 | 212.0 | 67.2 | 67.2 | 67.2 | 1,307.2 | 4.7 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 431.8 | 225.0 | 200.5 | 200.5 | 200.5 | 296.4 | 17.8 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 722.3 | 214.7 | 37.4 | 37.4 | 37.4 | 11,340.9 | 18.6 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,352.2 | 863.5 | 817.2 | 817.2 | 817.2 | 152.7 | 18.5 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 2,227.9 | 666.9 | 115.3 | 115.3 | 115.3 | 14,708.2 | 24.0 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 4,531.3 | 2,414.7 | 2,145.6 | 2,145.6 | 2,145.6 | 452.0 | 28.5 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 4,091.6 | 3,326.5 | 2,782.6 | 2,782.6 | 2,782.6 | 2,002.2 | 87.7 | 1 | passed | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018132021967816-affcd720](../../runs/20261018132021967816-affcd720/run.json) |

<a id="lfm2.5-230m-q4_0-cuda-default-ar"></a>
## lfm2.5-230m / akas / q4_0 / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json).

Peak device GiB: 7.2; peak host process RSS GiB: 1.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 9,633.1 | 974.0 | 957.3 | 957.3 | 236.2 | 268.2 | 2.2 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 9,573.1 | 809.2 | 345.1 | 345.1 | 345.1 | 213.9 | 1.2 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 504.2 | 1,094.6 | 713.1 | 713.1 | 713.1 | 253.9 | 3.7 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 6,845.6 | 1,157.2 | 313.0 | 313.0 | 313.0 | 1,196.7 | 3.5 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 34,510.1 | 4,105.1 | 4,017.0 | 4,017.0 | 4,017.0 | 14.8 | 3.9 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 26,057.8 | 2,736.5 | 1,024.0 | 1,024.0 | 1,024.0 | 1,257.5 | 5.8 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 12,264.9 | 13,412.7 | 10,598.0 | 10,598.0 | 10,598.0 | 167.0 | 5.9 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 49,075.6 | 3,762.9 | 1,695.8 | 1,695.8 | 1,695.8 | 2,670.8 | 17.2 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 19,193.5 | 19,713.3 | 15,783.8 | 15,783.8 | 15,783.8 | 426.8 | 15.5 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 47,825.1 | 3,871.0 | 1,692.4 | 1,692.4 | 1,692.4 | 10,962.6 | 67.6 | 1 | passed | [20261018132230646310-93a139e8](../../runs/20261018132230646310-93a139e8/run.json) |

<a id="lfm2.5-230m-q4_0-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / q4_0 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json).

Peak device GiB: 4.0; peak host process RSS GiB: 1.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 10,135.7 | 983.0 | 967.1 | 967.1 | 243.5 | 259.1 | 2.1 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 10,246.2 | 809.3 | 358.7 | 358.7 | 358.7 | 199.9 | 1.2 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 502.8 | 1,081.5 | 706.9 | 706.9 | 706.9 | 254.6 | 3.7 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 6,733.5 | 1,165.6 | 309.8 | 309.8 | 309.8 | 1,216.6 | 3.4 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 34,325.7 | 4,087.1 | 3,999.3 | 3,999.3 | 3,999.3 | 14.9 | 3.9 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 26,050.2 | 2,811.0 | 1,034.0 | 1,034.0 | 1,034.0 | 1,257.9 | 5.7 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 12,061.6 | 13,480.8 | 10,601.4 | 10,601.4 | 10,601.4 | 169.8 | 5.9 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 49,970.5 | 3,956.1 | 1,751.4 | 1,751.4 | 1,751.4 | 2,623.0 | 16.3 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 24,337.7 | 19,980.6 | 16,686.3 | 16,686.3 | 16,686.3 | 336.6 | 14.6 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 48,749.5 | 4,361.6 | 1,799.6 | 1,799.6 | 1,799.6 | 10,754.7 | 60.1 | 1 | passed | [20261018132342934575-aca09c85](../../runs/20261018132342934575-aca09c85/run.json) |

<a id="lfm2.5-230m-q4_0-webgpu-default-ar"></a>
## lfm2.5-230m / akas / q4_0 / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json).

Peak device GiB: 13.6; peak host process RSS GiB: 1.8. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "a60d051026a6a775", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,599.7 | 432.0 | 407.6 | 407.6 | 154.6 | 449.4 | 3.0 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,446.0 | 342.2 | 71.6 | 71.6 | 71.6 | 1,416.3 | 2.9 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 364.4 | 177.9 | 159.7 | 159.7 | 159.7 | 351.2 | 22.5 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 580.9 | 180.2 | 30.3 | 30.3 | 30.3 | 14,102.9 | 22.2 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 2,521.2 | 707.0 | 665.5 | 665.5 | 665.5 | 203.1 | 22.6 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 1,986.4 | 610.2 | 103.3 | 103.3 | 103.3 | 16,495.8 | 26.2 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,664.8 | 1,906.6 | 1,698.9 | 1,698.9 | 1,698.9 | 558.8 | 36.0 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 3,312.3 | 2,481.4 | 2,103.8 | 2,103.8 | 2,103.8 | 2,473.2 | 116.0 | 1 | passed | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018132454970187-edf9a076](../../runs/20261018132454970187-edf9a076/run.json) |

<a id="lfm2.5-230m-q4_0-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / q4_0 / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json).

Peak device GiB: 4.1; peak host process RSS GiB: 1.8. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "74d933f0b09e199a", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,600.0 | 442.4 | 416.9 | 416.9 | 196.0 | 260.6 | 3.1 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,348.3 | 313.7 | 66.5 | 66.5 | 66.5 | 1,518.9 | 3.2 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 353.5 | 182.3 | 162.6 | 162.6 | 162.6 | 362.1 | 21.9 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 561.4 | 175.7 | 29.3 | 29.3 | 29.3 | 14,591.0 | 22.8 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 2,514.1 | 705.1 | 663.7 | 663.7 | 663.7 | 203.7 | 22.7 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 1,826.5 | 571.7 | 95.3 | 95.3 | 95.3 | 17,940.2 | 28.0 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,652.0 | 1,892.4 | 1,687.1 | 1,687.1 | 1,687.1 | 560.8 | 36.3 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 3,318.8 | 2,470.9 | 2,097.0 | 2,097.0 | 2,097.0 | 2,468.3 | 116.3 | 1 | passed | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018132713680721-e2e2fd23](../../runs/20261018132713680721-e2e2fd23/run.json) |

<a id="lfm2.5-230m-q4_k_m-cuda-default-ar"></a>
## lfm2.5-230m / akas / q4_k_m / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json).

Peak device GiB: 7.2; peak host process RSS GiB: 1.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 8,547.2 | 894.9 | 878.7 | 878.7 | 222.5 | 285.2 | 2.3 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 8,046.2 | 744.8 | 301.1 | 301.1 | 301.1 | 254.5 | 1.3 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 479.6 | 932.5 | 630.8 | 630.8 | 630.8 | 266.9 | 4.3 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 5,394.8 | 1,028.8 | 254.4 | 254.4 | 254.4 | 1,518.5 | 3.9 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 27,570.3 | 3,577.7 | 3,491.7 | 3,491.7 | 3,491.7 | 18.6 | 4.5 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 20,828.8 | 2,547.7 | 863.8 | 863.8 | 863.8 | 1,573.2 | 6.3 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 10,512.9 | 11,642.6 | 9,174.2 | 9,174.2 | 9,174.2 | 194.8 | 6.8 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 39,640.4 | 3,635.8 | 1,478.2 | 1,478.2 | 1,478.2 | 3,306.5 | 17.8 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 20,518.2 | 17,639.6 | 14,613.0 | 14,613.0 | 14,613.0 | 399.3 | 16.7 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 38,824.6 | 3,734.6 | 1,475.4 | 1,475.4 | 1,475.4 | 13,504.0 | 70.1 | 1 | passed | [20261018132935164249-1ec690bc](../../runs/20261018132935164249-1ec690bc/run.json) |

<a id="lfm2.5-230m-q4_k_m-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / q4_k_m / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json).

Peak device GiB: 4.0; peak host process RSS GiB: 1.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 8,561.2 | 902.3 | 885.9 | 885.9 | 229.5 | 270.0 | 2.3 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 8,529.2 | 745.1 | 311.8 | 311.8 | 311.8 | 240.1 | 1.3 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 475.6 | 895.4 | 612.1 | 612.1 | 612.1 | 269.2 | 4.5 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 5,418.2 | 1,048.7 | 256.5 | 256.5 | 256.5 | 1,511.9 | 3.8 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 27,277.0 | 3,573.9 | 3,486.9 | 3,486.9 | 3,486.9 | 18.8 | 4.5 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 21,153.8 | 2,639.6 | 883.2 | 883.2 | 883.2 | 1,549.0 | 6.1 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 10,812.4 | 11,804.6 | 9,330.8 | 9,330.8 | 9,330.8 | 189.4 | 6.7 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 40,457.8 | 3,827.7 | 1,527.4 | 1,527.4 | 1,527.4 | 3,239.7 | 16.9 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 16,734.2 | 18,037.3 | 14,296.6 | 14,296.6 | 14,296.6 | 489.5 | 17.1 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 39,506.5 | 4,212.7 | 1,561.2 | 1,561.2 | 1,561.2 | 13,270.9 | 62.3 | 1 | passed | [20261018133054198297-56a5e435](../../runs/20261018133054198297-56a5e435/run.json) |

<a id="lfm2.5-230m-q4_k_m-webgpu-default-ar"></a>
## lfm2.5-230m / akas / q4_k_m / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json).

Peak device GiB: 13.6; peak host process RSS GiB: 1.8. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "a60d051026a6a775", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,449.3 | 301.7 | 288.9 | 288.9 | 116.8 | 455.7 | 5.0 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,319.0 | 241.7 | 61.6 | 61.6 | 61.6 | 1,552.6 | 4.1 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 272.1 | 167.3 | 146.0 | 146.0 | 146.0 | 470.4 | 23.9 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 527.0 | 167.4 | 27.6 | 27.6 | 27.6 | 15,545.7 | 23.9 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 2,294.7 | 645.5 | 607.6 | 607.6 | 607.6 | 223.1 | 24.8 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 1,831.6 | 565.4 | 95.3 | 95.3 | 95.3 | 17,890.3 | 28.3 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,318.7 | 1,773.2 | 1,575.1 | 1,575.1 | 1,575.1 | 617.1 | 38.8 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 3,022.7 | 2,330.6 | 1,966.8 | 1,966.8 | 1,966.8 | 2,710.2 | 124.0 | 1 | passed | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018133213394457-6422e80a](../../runs/20261018133213394457-6422e80a/run.json) |

<a id="lfm2.5-230m-q4_k_m-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / q4_k_m / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json).

Peak device GiB: 4.1; peak host process RSS GiB: 1.8. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "74d933f0b09e199a", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 1,452.0 | 281.3 | 270.3 | 270.3 | 155.7 | 278.8 | 4.3 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,238.5 | 236.1 | 58.4 | 58.4 | 58.4 | 1,653.6 | 4.2 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 322.4 | 168.6 | 150.1 | 150.1 | 150.1 | 397.1 | 23.7 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 513.9 | 164.4 | 26.9 | 26.9 | 26.9 | 15,939.9 | 24.3 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 2,276.4 | 642.5 | 604.6 | 604.6 | 604.6 | 224.9 | 24.9 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 1,703.1 | 538.5 | 89.0 | 89.0 | 89.0 | 19,240.2 | 29.7 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 2,903.8 | 1,777.7 | 1,552.2 | 1,552.2 | 1,552.2 | 705.3 | 39.4 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 3,045.3 | 2,346.8 | 1,980.7 | 1,980.7 | 1,980.7 | 2,690.1 | 123.1 | 1 | passed | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018133440998595-e60f24f2](../../runs/20261018133440998595-e60f24f2/run.json) |

<a id="qwen3.5-0.8b-bf16-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json).

Peak device GiB: 13.5; peak host process RSS GiB: 1.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "b4805ec444a95869", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,848.2 | 143.9 | 143.6 | 143.6 | 101.4 | 212.9 | 8.3 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 3,773.5 | 132.9 | 85.4 | 85.4 | 85.4 | 542.7 | 7.5 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 670.7 | 459.8 | 395.2 | 395.2 | 395.2 | 190.8 | 8.7 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 2,668.7 | 489.0 | 124.6 | 124.6 | 124.6 | 3,069.7 | 8.2 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 10,669.6 | 1,461.4 | 1,423.8 | 1,423.8 | 1,423.8 | 48.0 | 10.9 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 8,974.5 | 1,409.0 | 402.1 | 402.1 | 402.1 | 3,651.2 | 11.4 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 7,243.8 | 3,152.8 | 2,863.6 | 2,863.6 | 2,863.6 | 282.7 | 21.9 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 13,531.0 | 2,404.2 | 626.9 | 626.9 | 626.9 | 9,686.8 | 27.3 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 8,044.8 | 4,063.3 | 3,632.9 | 3,632.9 | 3,632.9 | 1,018.3 | 68.1 | 1 | passed | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018133711475982-66e2695e](../../runs/20261018133711475982-66e2695e/run.json) |

<a id="qwen3.5-0.8b-bf16-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json).

Peak device GiB: 16.3; peak host process RSS GiB: 1.7. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "3d7824fd78938e93", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 250.1 | 137.1 | 121.4 | 121.4 | 99.8 | 338.2 | 7.4 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,882.0 | 196.5 | 94.3 | 94.3 | 94.3 | 710.6 | 5.1 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 414.9 | 470.9 | 369.1 | 369.1 | 369.1 | 308.5 | 8.5 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 2,435.9 | 690.6 | 124.9 | 124.9 | 124.9 | 3,363.0 | 5.8 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 9,746.1 | 1,481.9 | 1,438.5 | 1,438.5 | 1,438.5 | 52.5 | 10.8 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 8,250.8 | 1,741.3 | 398.6 | 398.6 | 398.6 | 3,971.5 | 9.2 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 4,456.0 | 3,959.8 | 3,260.9 | 3,260.9 | 3,260.9 | 459.6 | 19.2 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 12,589.0 | 3,046.6 | 626.3 | 626.3 | 626.3 | 10,411.6 | 21.9 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 6,199.5 | 4,656.7 | 3,946.4 | 3,946.4 | 3,946.4 | 1,321.4 | 63.6 | 1 | passed | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018133859638625-0eef8c58](../../runs/20261018133859638625-0eef8c58/run.json) |

<a id="qwen3.5-0.8b-bf16-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json).

Peak device GiB: 10.4; peak host process RSS GiB: 1.4. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "fd5803b380a76562", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,786.3 | 131.5 | 131.4 | 131.4 | 88.2 | 199.8 | 9.8 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,183.7 | 47.6 | 35.5 | 35.5 | 35.5 | 937.9 | 21.0 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 677.0 | 426.8 | 371.2 | 371.2 | 371.2 | 189.1 | 9.4 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 2,326.8 | 181.9 | 81.1 | 81.1 | 81.1 | 3,520.7 | 22.0 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 10,911.9 | 1,304.6 | 1,276.4 | 1,276.4 | 1,276.4 | 46.9 | 12.3 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 7,255.9 | 661.8 | 270.0 | 270.0 | 270.0 | 4,516.0 | 24.2 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 7,135.4 | 2,897.2 | 2,649.0 | 2,649.0 | 2,649.0 | 287.0 | 23.7 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 9,543.5 | 1,046.4 | 381.0 | 381.0 | 381.0 | 13,734.1 | 62.5 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 8,753.4 | 3,755.9 | 3,416.2 | 3,416.2 | 3,416.2 | 935.9 | 72.4 | 1 | passed | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134119261822-597c8bda](../../runs/20261018134119261822-597c8bda/run.json) |

<a id="qwen3.5-0.8b-bf16-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json).

Peak device GiB: 12.7; peak host process RSS GiB: 1.7. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "072957e893c43afb", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 265.8 | 130.5 | 117.0 | 117.0 | 96.1 | 331.6 | 7.9 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,747.0 | 77.0 | 45.4 | 45.4 | 45.4 | 1,172.3 | 13.0 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 419.0 | 447.0 | 355.1 | 355.1 | 355.1 | 305.5 | 8.9 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 2,115.1 | 283.6 | 90.4 | 90.4 | 90.4 | 3,873.1 | 14.1 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 10,080.4 | 1,379.8 | 1,344.3 | 1,344.3 | 1,344.3 | 50.8 | 11.6 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 6,600.1 | 867.5 | 280.3 | 280.3 | 280.3 | 4,964.8 | 18.4 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 4,471.4 | 3,723.0 | 3,101.6 | 3,101.6 | 3,101.6 | 458.0 | 20.2 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 8,569.9 | 1,520.8 | 396.9 | 396.9 | 396.9 | 15,294.5 | 44.1 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 6,139.3 | 4,420.6 | 3,771.2 | 3,771.2 | 3,771.2 | 1,334.4 | 66.7 | 1 | passed | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134325570980-469e5ae3](../../runs/20261018134325570980-469e5ae3/run.json) |

<a id="qwen3.5-0.8b-bf16-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json).

Peak device GiB: 13.4; peak host process RSS GiB: 3.5. Loading and in-process compilation are included in host RSS.

fused_head is a native CUDA BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "0e041632c019cec9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 252.6 | 71.4 | 67.2 | 67.2 | 49.4 | 727.6 | 14.7 | 1 | passed | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 237.1 | 41.0 | 10.9 | 10.9 | 10.9 | 8,638.6 | 24.4 | 1 | passed | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 49.2 | 37.3 | 31.6 | 31.6 | 31.6 | 2,599.9 | 107.1 | 1 | passed | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 201.2 | 144.4 | 123.2 | 123.2 | 123.2 | 2,544.9 | 110.8 | 1 | passed | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134601703707-d59fcdfa](../../runs/20261018134601703707-d59fcdfa/run.json) |

<a id="qwen3.5-0.8b-bf16-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json).

Peak device GiB: 15.1; peak host process RSS GiB: 3.7. Loading and in-process compilation are included in host RSS.

fused_head is a native CUDA BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "d663818e41f437c9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 154.6 | 15.4 | 15.2 | 15.2 | 15.0 | 710.2 | 61.7 | 1 | passed | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 219.4 | 15.9 | 7.4 | 7.4 | 7.4 | 9,333.0 | 63.1 | 1 | passed | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 45.8 | 53.8 | 41.8 | 41.8 | 41.8 | 2,797.7 | 74.4 | 1 | passed | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 187.6 | 139.4 | 118.3 | 118.3 | 118.3 | 2,729.5 | 114.8 | 1 | passed | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018134847189946-c5bec17a](../../runs/20261018134847189946-c5bec17a/run.json) |

<a id="qwen3.5-0.8b-bf16-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json).

Peak device GiB: 10.4; peak host process RSS GiB: 3.6. Loading and in-process compilation are included in host RSS.

fused_head is a native CUDA BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "d1f846ef7dd64675", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 252.8 | 70.0 | 65.9 | 65.9 | 50.6 | 588.2 | 15.3 | 1 | passed | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 233.0 | 31.6 | 10.0 | 10.0 | 10.0 | 8,788.4 | 31.6 | 1 | passed | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 49.5 | 37.2 | 31.5 | 31.5 | 31.5 | 2,587.6 | 107.5 | 1 | passed | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 197.5 | 142.6 | 121.6 | 121.6 | 121.6 | 2,592.2 | 112.2 | 1 | passed | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135254888257-4722a5b7](../../runs/20261018135254888257-4722a5b7/run.json) |

<a id="qwen3.5-0.8b-bf16-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json).

Peak device GiB: 11.5; peak host process RSS GiB: 3.7. Loading and in-process compilation are included in host RSS.

fused_head is a native CUDA BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "ca10f00dab2b2095", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 165.1 | 16.2 | 15.9 | 15.9 | 15.0 | 646.5 | 62.0 | 1 | passed | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 215.5 | 14.8 | 7.1 | 7.1 | 7.1 | 9,501.4 | 67.4 | 1 | passed | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 44.9 | 55.3 | 42.5 | 42.5 | 42.5 | 2,850.5 | 72.4 | 1 | passed | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 188.3 | 133.2 | 113.9 | 113.9 | 113.9 | 2,719.7 | 120.2 | 1 | passed | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135534665929-102701fc](../../runs/20261018135534665929-102701fc/run.json) |

<a id="qwen3.5-0.8b-f16-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / f16 / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json).

Peak device GiB: 13.5; peak host process RSS GiB: 4.1. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "b4805ec444a95869", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 2,488.1 | 145.2 | 144.3 | 144.3 | 94.5 | 204.3 | 9.1 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,478.7 | 133.1 | 71.9 | 71.9 | 71.9 | 826.3 | 7.5 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 701.2 | 302.3 | 274.8 | 274.8 | 274.8 | 182.5 | 13.2 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,964.3 | 311.5 | 88.3 | 88.3 | 88.3 | 4,170.5 | 12.8 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 7,771.4 | 972.9 | 950.6 | 950.6 | 950.6 | 65.9 | 16.4 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 6,899.3 | 990.5 | 301.1 | 301.1 | 301.1 | 4,749.4 | 16.2 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 6,553.5 | 2,535.1 | 2,328.1 | 2,328.1 | 2,328.1 | 312.5 | 26.9 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 11,108.8 | 2,028.3 | 518.3 | 518.3 | 518.3 | 11,798.9 | 32.3 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 7,403.3 | 3,615.7 | 3,244.9 | 3,244.9 | 3,244.9 | 1,106.5 | 76.0 | 1 | passed | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018135933378298-423b61d0](../../runs/20261018135933378298-423b61d0/run.json) |

<a id="qwen3.5-0.8b-f16-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / f16 / cuda / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json).

Peak device GiB: 16.2; peak host process RSS GiB: 4.4. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "3d7824fd78938e93", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 315.2 | 105.3 | 97.9 | 97.9 | 82.2 | 311.8 | 9.8 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,058.2 | 134.1 | 65.9 | 65.9 | 65.9 | 995.0 | 7.5 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 308.8 | 383.4 | 294.3 | 294.3 | 294.3 | 414.6 | 10.4 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,828.7 | 466.8 | 92.0 | 92.0 | 92.0 | 4,479.7 | 8.6 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 7,151.2 | 1,139.4 | 1,104.1 | 1,104.1 | 1,104.1 | 71.6 | 14.0 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 6,450.2 | 1,226.6 | 304.0 | 304.0 | 304.0 | 5,080.2 | 13.0 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 4,288.7 | 3,224.3 | 2,732.1 | 2,732.1 | 2,732.1 | 477.5 | 22.8 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 10,349.6 | 2,597.2 | 518.7 | 518.7 | 518.7 | 12,664.5 | 25.8 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 5,303.0 | 3,990.5 | 3,380.9 | 3,380.9 | 3,380.9 | 1,544.8 | 74.3 | 1 | passed | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018140456927334-1cf13a3a](../../runs/20261018140456927334-1cf13a3a/run.json) |

<a id="qwen3.5-0.8b-f16-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / f16 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json).

Peak device GiB: 10.4; peak host process RSS GiB: 4.1. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "fd5803b380a76562", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 2,461.9 | 132.7 | 131.9 | 131.9 | 97.5 | 197.9 | 8.8 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,666.5 | 47.7 | 32.9 | 32.9 | 32.9 | 1,228.9 | 20.9 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 701.7 | 289.5 | 264.3 | 264.3 | 264.3 | 182.4 | 13.8 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,782.3 | 150.1 | 64.1 | 64.1 | 64.1 | 4,596.3 | 26.7 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 8,185.9 | 990.1 | 968.4 | 968.4 | 968.4 | 62.5 | 16.2 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 5,868.3 | 554.1 | 221.4 | 221.4 | 221.4 | 5,583.9 | 28.9 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 6,474.5 | 2,388.1 | 2,202.2 | 2,202.2 | 2,202.2 | 316.3 | 28.5 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 8,263.2 | 970.2 | 338.0 | 338.0 | 338.0 | 15,862.1 | 67.5 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 7,333.2 | 3,383.4 | 3,054.9 | 3,054.9 | 3,054.9 | 1,117.1 | 80.8 | 1 | passed | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018141112684266-3eb5b891](../../runs/20261018141112684266-3eb5b891/run.json) |

<a id="qwen3.5-0.8b-f16-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / f16 / cuda / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json).

Peak device GiB: 12.6; peak host process RSS GiB: 4.4. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "072957e893c43afb", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 310.2 | 102.6 | 95.5 | 95.5 | 73.7 | 307.9 | 11.2 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,396.0 | 65.0 | 37.4 | 37.4 | 37.4 | 1,467.1 | 15.4 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 414.7 | 371.0 | 305.1 | 305.1 | 305.1 | 308.7 | 10.8 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,631.7 | 235.5 | 71.3 | 71.3 | 71.3 | 5,020.6 | 17.0 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 7,421.0 | 1,083.7 | 1,053.5 | 1,053.5 | 1,053.5 | 69.0 | 14.8 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 5,360.9 | 709.0 | 228.1 | 228.1 | 228.1 | 6,112.4 | 22.6 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,476.9 | 3,076.9 | 2,535.7 | 2,535.7 | 2,535.7 | 589.0 | 24.6 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 7,490.9 | 1,392.4 | 351.1 | 351.1 | 351.1 | 17,497.6 | 48.2 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 5,237.0 | 3,821.9 | 3,253.7 | 3,253.7 | 3,253.7 | 1,564.3 | 77.2 | 1 | passed | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018141518118717-e1387382](../../runs/20261018141518118717-e1387382/run.json) |

<a id="qwen3.5-0.8b-f16-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json).

Peak device GiB: 13.4; peak host process RSS GiB: 6.1. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "0e041632c019cec9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 251.6 | 69.8 | 65.8 | 65.8 | 51.2 | 656.2 | 14.5 | 1 | passed | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 237.1 | 41.1 | 10.9 | 10.9 | 10.9 | 8,638.3 | 24.3 | 1 | passed | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 49.9 | 37.5 | 31.8 | 31.8 | 31.8 | 2,566.2 | 106.7 | 1 | passed | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 199.9 | 144.4 | 123.1 | 123.1 | 123.1 | 2,560.8 | 110.8 | 1 | passed | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142118202879-02440bd9](../../runs/20261018142118202879-02440bd9/run.json) |

<a id="qwen3.5-0.8b-f16-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json).

Peak device GiB: 15.1; peak host process RSS GiB: 6.2. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "d663818e41f437c9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 161.8 | 16.3 | 16.0 | 16.0 | 14.7 | 880.7 | 61.8 | 1 | passed | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 220.1 | 15.8 | 7.4 | 7.4 | 7.4 | 9,304.5 | 63.1 | 1 | passed | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 46.5 | 55.8 | 43.2 | 43.2 | 43.2 | 2,753.1 | 71.7 | 1 | passed | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 189.1 | 139.9 | 118.8 | 118.8 | 118.8 | 2,707.4 | 114.4 | 1 | passed | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142401278788-e1b20739](../../runs/20261018142401278788-e1b20739/run.json) |

<a id="qwen3.5-0.8b-f16-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json).

Peak device GiB: 10.4; peak host process RSS GiB: 6.1. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "d1f846ef7dd64675", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 252.6 | 72.1 | 67.8 | 67.8 | 48.7 | 738.0 | 14.9 | 1 | passed | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 233.3 | 32.0 | 10.0 | 10.0 | 10.0 | 8,779.0 | 31.2 | 1 | passed | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 47.1 | 37.3 | 31.4 | 31.4 | 31.4 | 2,714.8 | 107.2 | 1 | passed | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 197.9 | 143.9 | 122.6 | 122.6 | 122.6 | 2,586.7 | 111.2 | 1 | passed | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018142738044876-56415fb5](../../runs/20261018142738044876-56415fb5/run.json) |

<a id="qwen3.5-0.8b-f16-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json).

Peak device GiB: 11.5; peak host process RSS GiB: 6.2. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "ca10f00dab2b2095", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 109.6 | 16.3 | 15.8 | 15.8 | 15.1 | 645.0 | 61.8 | 1 | passed | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 216.6 | 14.9 | 7.1 | 7.1 | 7.1 | 9,454.0 | 67.1 | 1 | passed | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 46.6 | 55.8 | 43.2 | 43.2 | 43.2 | 2,747.7 | 71.7 | 1 | passed | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 191.6 | 141.9 | 120.5 | 120.5 | 120.5 | 2,671.8 | 112.8 | 1 | passed | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143024873596-fc39ef42](../../runs/20261018143024873596-fc39ef42/run.json) |

<a id="qwen3.5-0.8b-q4_0-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json).

Peak device GiB: 12.5; peak host process RSS GiB: 2.2. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "b4805ec444a95869", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,201.4 | 227.4 | 225.1 | 225.1 | 136.9 | 209.4 | 5.7 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 3,152.8 | 201.5 | 100.0 | 100.0 | 100.0 | 649.6 | 5.0 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 545.2 | 389.0 | 332.3 | 332.3 | 332.3 | 234.8 | 10.3 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,447.6 | 410.8 | 74.2 | 74.2 | 74.2 | 5,659.2 | 9.7 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 5,843.5 | 1,283.9 | 1,226.1 | 1,226.1 | 1,226.1 | 87.6 | 12.5 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 5,266.2 | 1,237.2 | 260.4 | 260.4 | 260.4 | 6,222.3 | 12.9 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 5,516.7 | 2,607.4 | 2,348.2 | 2,348.2 | 2,348.2 | 371.2 | 26.5 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 8,872.7 | 2,072.2 | 438.2 | 438.2 | 438.2 | 14,772.4 | 31.8 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 5,539.3 | 3,043.4 | 2,694.4 | 2,694.4 | 2,694.4 | 1,478.9 | 90.7 | 1 | passed | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143407909404-6fe39872](../../runs/20261018143407909404-6fe39872/run.json) |

<a id="qwen3.5-0.8b-q4_0-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json).

Peak device GiB: 15.2; peak host process RSS GiB: 2.5. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "3d7824fd78938e93", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 271.8 | 133.7 | 119.9 | 119.9 | 87.9 | 465.1 | 7.8 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,726.6 | 144.4 | 78.5 | 78.5 | 78.5 | 751.1 | 6.9 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 369.3 | 470.3 | 358.8 | 358.8 | 358.8 | 346.6 | 8.5 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,363.4 | 601.5 | 74.7 | 74.7 | 74.7 | 6,008.4 | 6.6 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 5,430.1 | 1,243.0 | 1,184.5 | 1,184.5 | 1,184.5 | 94.3 | 12.9 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 4,970.2 | 1,401.0 | 254.6 | 254.6 | 254.6 | 6,592.9 | 11.4 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,718.2 | 2,737.6 | 2,327.4 | 2,327.4 | 2,327.4 | 550.8 | 26.9 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 8,335.5 | 2,242.9 | 423.4 | 423.4 | 423.4 | 15,724.5 | 29.9 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 4,236.3 | 3,077.3 | 2,621.7 | 2,621.7 | 2,621.7 | 1,933.8 | 96.9 | 1 | passed | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018143824470622-2fc170e3](../../runs/20261018143824470622-2fc170e3/run.json) |

<a id="qwen3.5-0.8b-q4_0-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json).

Peak device GiB: 9.4; peak host process RSS GiB: 2.2. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "fd5803b380a76562", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 3,162.6 | 200.6 | 199.0 | 199.0 | 126.8 | 206.8 | 6.3 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,947.3 | 55.9 | 38.5 | 38.5 | 38.5 | 1,051.7 | 17.9 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 547.3 | 367.3 | 316.7 | 316.7 | 316.7 | 233.9 | 10.9 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,326.9 | 170.1 | 55.9 | 55.9 | 55.9 | 6,173.7 | 23.5 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 5,861.3 | 1,225.4 | 1,173.2 | 1,173.2 | 1,173.2 | 87.4 | 13.1 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 4,617.3 | 627.5 | 198.2 | 198.2 | 198.2 | 7,096.7 | 25.5 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 4,342.8 | 2,446.8 | 2,159.5 | 2,159.5 | 2,159.5 | 471.6 | 28.9 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 6,963.0 | 989.8 | 303.0 | 303.0 | 303.0 | 18,824.0 | 66.3 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 5,874.1 | 2,868.5 | 2,574.3 | 2,574.3 | 2,574.3 | 1,394.6 | 95.8 | 1 | passed | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144310958228-59226851](../../runs/20261018144310958228-59226851/run.json) |

<a id="qwen3.5-0.8b-q4_0-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json).

Peak device GiB: 11.6; peak host process RSS GiB: 2.5. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "072957e893c43afb", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 271.2 | 114.5 | 104.3 | 104.3 | 86.9 | 377.5 | 8.6 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,673.9 | 58.4 | 37.7 | 37.7 | 37.7 | 1,223.5 | 17.1 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 362.1 | 444.2 | 342.0 | 342.0 | 342.0 | 353.5 | 9.0 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 1,221.8 | 269.1 | 59.6 | 59.6 | 59.6 | 6,704.6 | 14.9 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 5,445.7 | 1,290.0 | 1,226.9 | 1,226.9 | 1,226.9 | 94.0 | 12.4 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 4,201.3 | 749.2 | 194.8 | 194.8 | 194.8 | 7,799.6 | 21.4 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,573.1 | 2,524.9 | 2,160.2 | 2,160.2 | 2,160.2 | 573.2 | 29.0 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 6,351.3 | 1,261.7 | 302.5 | 302.5 | 302.5 | 20,637.0 | 52.5 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 3,842.1 | 2,927.3 | 2,475.2 | 2,475.2 | 2,475.2 | 2,132.2 | 101.7 | 1 | passed | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144546940645-b9caf87b](../../runs/20261018144546940645-b9caf87b/run.json) |

<a id="qwen3.5-0.8b-q4_0-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json).

Peak device GiB: 12.3; peak host process RSS GiB: 2.9. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "0e041632c019cec9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 257.0 | 81.6 | 76.2 | 76.2 | 56.9 | 527.2 | 13.5 | 1 | passed | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 240.7 | 45.9 | 11.4 | 11.4 | 11.4 | 8,508.2 | 21.8 | 1 | passed | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 47.4 | 37.7 | 31.7 | 31.7 | 31.7 | 2,699.5 | 106.0 | 1 | passed | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 194.4 | 143.8 | 122.1 | 122.1 | 122.1 | 2,633.4 | 111.3 | 1 | passed | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018144907492049-2f845f93](../../runs/20261018144907492049-2f845f93/run.json) |

<a id="qwen3.5-0.8b-q4_0-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json).

Peak device GiB: 14.0; peak host process RSS GiB: 2.9. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "d663818e41f437c9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 127.6 | 16.6 | 16.2 | 16.2 | 16.3 | 618.2 | 57.1 | 1 | passed | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 223.8 | 13.1 | 6.8 | 6.8 | 6.8 | 9,150.8 | 76.6 | 1 | passed | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 43.2 | 54.3 | 41.5 | 41.5 | 41.5 | 2,963.1 | 73.7 | 1 | passed | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 183.0 | 137.1 | 116.3 | 116.3 | 116.3 | 2,797.2 | 116.7 | 1 | passed | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145209029629-85607bc4](../../runs/20261018145209029629-85607bc4/run.json) |

<a id="qwen3.5-0.8b-q4_0-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json).

Peak device GiB: 9.3; peak host process RSS GiB: 2.8. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "d1f846ef7dd64675", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 253.5 | 78.5 | 73.4 | 73.4 | 57.2 | 469.7 | 13.9 | 1 | passed | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 236.8 | 33.8 | 10.3 | 10.3 | 10.3 | 8,647.4 | 29.6 | 1 | passed | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 48.3 | 38.8 | 32.5 | 32.5 | 32.5 | 2,648.6 | 103.2 | 1 | passed | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 195.8 | 149.9 | 126.6 | 126.6 | 126.6 | 2,614.5 | 106.7 | 1 | passed | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145559662873-743978c8](../../runs/20261018145559662873-743978c8/run.json) |

<a id="qwen3.5-0.8b-q4_0-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json).

Peak device GiB: 10.5; peak host process RSS GiB: 2.9. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "ca10f00dab2b2095", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 163.8 | 17.3 | 17.0 | 17.0 | 15.8 | 705.5 | 58.2 | 1 | passed | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 219.5 | 12.6 | 6.6 | 6.6 | 6.6 | 9,328.8 | 79.5 | 1 | passed | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 44.5 | 57.5 | 43.7 | 43.7 | 43.7 | 2,878.1 | 69.6 | 1 | passed | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 182.6 | 140.7 | 118.8 | 118.8 | 118.8 | 2,803.4 | 113.7 | 1 | passed | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018145840104186-24d4d259](../../runs/20261018145840104186-24d4d259/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json).

Peak device GiB: 12.5; peak host process RSS GiB: 2.3. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "b4805ec444a95869", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 2,192.0 | 160.8 | 159.2 | 159.2 | 92.8 | 248.7 | 8.9 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,145.5 | 146.4 | 70.2 | 70.2 | 70.2 | 954.6 | 6.8 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 394.0 | 262.8 | 226.8 | 226.8 | 226.8 | 324.9 | 15.2 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 805.7 | 275.3 | 42.6 | 42.6 | 42.6 | 10,167.9 | 14.5 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,215.8 | 873.6 | 824.1 | 824.1 | 824.1 | 159.2 | 18.3 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 3,054.8 | 900.6 | 157.7 | 157.7 | 157.7 | 10,726.9 | 17.8 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,707.1 | 1,985.1 | 1,762.9 | 1,762.9 | 1,762.9 | 552.5 | 35.0 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 5,476.6 | 1,635.9 | 283.4 | 283.4 | 283.4 | 23,933.0 | 40.7 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 3,859.3 | 2,375.3 | 2,072.6 | 2,072.6 | 2,072.6 | 2,122.7 | 118.7 | 1 | passed | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018150214725744-e6b064ad](../../runs/20261018150214725744-e6b064ad/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json).

Peak device GiB: 15.2; peak host process RSS GiB: 2.6. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "3d7824fd78938e93", "model_options": {"fused_head": false, "kv_dtype": "bfloat16"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 248.0 | 89.4 | 82.6 | 82.6 | 65.1 | 369.2 | 12.6 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,891.2 | 123.0 | 60.5 | 60.5 | 60.5 | 1,082.9 | 8.1 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 271.5 | 353.6 | 268.3 | 268.3 | 268.3 | 471.5 | 11.3 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 774.3 | 446.5 | 43.7 | 43.7 | 43.7 | 10,579.9 | 9.0 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,100.9 | 1,066.1 | 988.9 | 988.9 | 988.9 | 165.1 | 15.0 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 2,937.6 | 1,207.3 | 159.5 | 159.5 | 159.5 | 11,154.6 | 13.3 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 2,346.1 | 2,181.8 | 1,781.5 | 1,781.5 | 1,781.5 | 872.9 | 34.5 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 5,240.3 | 1,829.5 | 278.1 | 278.1 | 278.1 | 25,012.1 | 37.1 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 2,922.4 | 2,473.5 | 2,054.7 | 2,054.7 | 2,054.7 | 2,803.2 | 122.0 | 1 | passed | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018150729812014-9634f119](../../runs/20261018150729812014-9634f119/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json).

Peak device GiB: 9.5; peak host process RSS GiB: 2.2. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "fd5803b380a76562", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 2,167.8 | 146.7 | 145.4 | 145.4 | 90.4 | 373.6 | 8.2 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,504.1 | 51.8 | 33.6 | 33.6 | 33.6 | 1,361.6 | 19.3 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 392.4 | 239.3 | 209.1 | 209.1 | 209.1 | 326.2 | 16.7 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 765.3 | 141.9 | 35.8 | 35.8 | 35.8 | 10,704.1 | 28.2 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,210.1 | 891.4 | 839.6 | 839.6 | 839.6 | 159.5 | 17.9 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 2,808.8 | 528.6 | 132.0 | 132.0 | 132.0 | 11,666.1 | 30.3 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 3,770.5 | 1,908.0 | 1,705.5 | 1,705.5 | 1,705.5 | 543.2 | 36.2 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 3,870.9 | 2,282.4 | 2,002.8 | 2,002.8 | 2,002.8 | 2,116.3 | 122.8 | 1 | passed | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151237223272-af03ee15](../../runs/20261018151237223272-af03ee15/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json).

Peak device GiB: 11.7; peak host process RSS GiB: 2.6. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "072957e893c43afb", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 223.3 | 87.1 | 79.9 | 79.9 | 69.0 | 365.8 | 11.7 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 1,324.7 | 61.9 | 35.6 | 35.6 | 35.6 | 1,546.0 | 16.1 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 228.6 | 342.6 | 250.7 | 250.7 | 250.7 | 559.9 | 11.7 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 735.5 | 235.7 | 38.5 | 38.5 | 38.5 | 11,138.2 | 17.0 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 3,096.0 | 1,023.7 | 952.4 | 952.4 | 952.4 | 165.4 | 15.6 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 2,681.6 | 702.2 | 135.5 | 135.5 | 135.5 | 12,219.4 | 22.8 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 2,760.4 | 2,122.9 | 1,792.3 | 1,792.3 | 1,792.3 | 741.9 | 34.6 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 2,904.1 | 2,405.6 | 2,005.9 | 2,005.9 | 2,005.9 | 2,820.8 | 125.1 | 1 | passed | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151512909614-9b31261c](../../runs/20261018151512909614-9b31261c/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json).

Peak device GiB: 12.4; peak host process RSS GiB: 2.9. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "0e041632c019cec9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 253.1 | 73.0 | 68.6 | 68.6 | 51.8 | 540.4 | 15.2 | 1 | passed | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 237.4 | 42.6 | 11.0 | 11.0 | 11.0 | 8,627.0 | 23.5 | 1 | passed | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 46.0 | 37.1 | 31.0 | 31.0 | 31.0 | 2,785.3 | 107.9 | 1 | passed | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 194.7 | 144.0 | 122.4 | 122.4 | 122.4 | 2,630.0 | 111.1 | 1 | passed | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018151836127904-b7ab440e](../../runs/20261018151836127904-b7ab440e/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / default / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json).

Peak device GiB: 14.1; peak host process RSS GiB: 3.0. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "d663818e41f437c9", "model_options": {"fused_head": false, "kv_dtype": "float16"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 99.1 | 15.3 | 14.8 | 14.8 | 15.1 | 625.6 | 61.9 | 1 | passed | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 221.9 | 16.1 | 7.5 | 7.5 | 7.5 | 9,229.4 | 62.1 | 1 | passed | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 44.6 | 53.1 | 41.2 | 41.2 | 41.2 | 2,866.9 | 75.3 | 1 | passed | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 184.3 | 140.8 | 119.0 | 119.0 | 119.0 | 2,778.0 | 113.7 | 1 | passed | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152156367296-42537605](../../runs/20261018152156367296-42537605/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / fp8 / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json).

Peak device GiB: 9.4; peak host process RSS GiB: 2.9. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "d1f846ef7dd64675", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 252.6 | 71.2 | 67.0 | 67.0 | 51.4 | 489.3 | 15.8 | 1 | passed | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 234.4 | 32.4 | 10.1 | 10.1 | 10.1 | 8,736.2 | 30.8 | 1 | passed | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 47.3 | 38.2 | 32.0 | 32.0 | 32.0 | 2,703.4 | 104.6 | 1 | passed | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 190.0 | 146.6 | 123.7 | 123.7 | 123.7 | 2,695.0 | 109.2 | 1 | passed | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152555702528-5964692d](../../runs/20261018152555702528-5964692d/run.json) |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / fp8 / mtp

Profile status: **complete**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json).

Peak device GiB: 10.6; peak host process RSS GiB: 2.9. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": "akas.models.qwen35.mtp:MTPHead", "id": "ca10f00dab2b2095", "model_options": {"fused_head": false, "kv_dtype": "fp8"}, "policy": {}, "prefill_requests": 32, "provider": "webgpu", "target": "webgpu-portable-v1", "trials": 1, "window": 2}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 159.3 | 15.3 | 15.1 | 15.1 | 14.3 | 554.0 | 66.1 | 1 | passed | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 217.7 | 15.0 | 7.2 | 7.2 | 7.2 | 9,406.3 | 66.6 | 1 | passed | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 44.0 | 55.6 | 42.5 | 42.5 | 42.5 | 2,907.9 | 71.9 | 1 | passed | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 182.3 | 137.6 | 116.5 | 116.5 | 116.5 | 2,808.1 | 116.3 | 1 | passed | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | — | — | — | — | — | — | — | 0 | unqualified | [20261018152842513121-97464fc5](../../runs/20261018152842513121-97464fc5/run.json) |

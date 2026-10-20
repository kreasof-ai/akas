# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; the measured LFM floating path executes FP16. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-24b-a2b-q4_0-cuda-default-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / cuda / default / ar

Profile status: **complete**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json).

Peak device GiB: 15.8; peak host process RSS GiB: 3.4. Loading and in-process compilation are included in host RSS.

ValueError: model logits differ from qualified baseline: b1-p512; ValueError: greedy output differs from baseline for B1/P32768

Selected recipe: `{"chunk": 512, "deferred": false, "graphs": true, "head": null, "id": "ee9b3acb399892d0", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 1, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 428.7 | 88.0 | 84.4 | 79.4 | 82.5 | 75.0 | 11.6 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 2,010.2 | 87.4 | 75.1 | 71.0 | 73.1 | 257.1 | 11.7 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 2,039.8 | 85.2 | 51.2 | 48.2 | 49.7 | 1,041.7 | 12.1 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 1,988.9 | 77.2 | 22.2 | 20.8 | 21.5 | 4,263.6 | 13.3 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 1,898.1 | 67.9 | 12.2 | 11.6 | 11.9 | 8,830.7 | 15.0 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 1,784.2 | 55.3 | 6.2 | 6.0 | 6.1 | 18,769.4 | 18.2 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 1,606.0 | 41.1 | 2.9 | 2.8 | 2.9 | 41,725.5 | 24.7 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 1,364.3 | 27.2 | 1.3 | 1.2 | 1.2 | 98,469.8 | 37.2 | 3 | passed | [20261019162711688563-1782af6e](../../runs/20261019162711688563-1782af6e/run.json) |

<a id="lfm2.5-24b-a2b-q4_0-webgpu-default-ar"></a>
## lfm2.5-24b-a2b / akas / q4_0 / webgpu / default / ar

Profile status: **interrupted**; qualification: **not_measured**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

TimeoutError:

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 113.2 | 41.8 | 38.5 | 34.7 | 36.5 | 291.1 | 25.3 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |
| 1 | 512 | 128 | native-token-step-v2 | 293.6 | 41.2 | 26.5 | 24.4 | 25.5 | 1,761.3 | 25.7 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 289.6 | 39.7 | 12.5 | 11.7 | 12.0 | 7,273.6 | 26.7 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |
| 1 | 8192 | 128 | native-token-step-v2 | 266.1 | 35.3 | 3.7 | 3.6 | 3.7 | 30,981.0 | 29.3 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |
| 1 | 16384 | 128 | native-token-step-v2 | 243.2 | 31.2 | 1.8 | 1.8 | 1.8 | 68,876.6 | 33.4 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |
| 1 | 32768 | 128 | native-token-step-v2 | 211.2 | 25.5 | 0.8 | 0.8 | 0.8 | 157,782.5 | 40.2 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |
| 1 | 65536 | 128 | native-token-step-v2 | 166.7 | 18.9 | 0.3 | 0.3 | 0.3 | 399,880.8 | 54.8 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |
| 1 | 130944 | 128 | native-token-step-v2 | 115.2 | 12.2 | 0.1 | — | 0.1 | 1,148,086.6 | 83.3 | 3 | unqualified | [20261019235748301427-fdb5ef2b](../../runs/20261019235748301427-fdb5ef2b/run.json) |

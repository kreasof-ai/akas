# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="lfm2.5-230m-bf16-cuda-default-ar"></a>
## lfm2.5-230m / akas / bf16 / cuda / default / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json).

Peak device GiB: 7.6; peak host process RSS GiB: 1.9. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded; ValueError: model logits differ from qualified baseline: b1-p2048; ValueError: model logits differ from qualified baseline: b4-p2048; ValueError: model logits differ from qualified baseline: b16-p2048; ValueError: model logits differ from qualified baseline: b64-p2048

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 9,330.4 | 479.7 | 477.3 | 437.7 | 424.1 | 4.4 | 2.3 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 10,603.7 | 431.3 | 262.7 | 226.4 | 224.8 | 243.6 | 2.6 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 9,505.3 | 1,784.2 | 1,718.3 | 1,580.1 | 1,486.2 | 21.1 | 2.6 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 8,852.7 | 1,331.9 | 391.6 | 345.6 | 344.1 | 1,076.9 | 3.4 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 37,055.6 | 3,778.1 | 3,712.9 | 3,524.8 | 3,405.8 | 23.4 | 4.6 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 32,902.0 | 2,233.3 | 1,073.5 | 992.1 | 989.0 | 1,128.8 | 7.5 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 72,216.4 | 13,129.8 | 12,653.5 | 11,783.3 | 11,087.4 | 54.4 | 5.5 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 58,701.7 | 3,464.7 | 1,789.1 | 1,743.1 | 1,722.4 | 2,447.3 | 18.6 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 66,450.6 | 22,320.8 | 20,702.8 | 19,717.3 | 17,372.1 | 372.2 | 12.7 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 58,499.8 | 3,686.0 | 1,842.7 | 1,806.7 | 1,779.4 | 10,060.9 | 67.7 | 3 | passed | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |

<a id="lfm2.5-230m-bf16-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / bf16 / cuda / fp8 / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json).

Peak device GiB: 4.4; peak host process RSS GiB: 1.8. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded; ValueError: deployment HTTP output differs from qualified model execution; ValueError: model logits differ from qualified baseline: b1-p32; ValueError: model logits differ from qualified baseline: b4-p32; ValueError: model logits differ from qualified baseline: b16-p2048; ValueError: model logits differ from qualified baseline: b64-p2048

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 9,298.8 | 480.0 | 477.6 | 437.7 | 425.0 | 4.5 | 2.3 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 11,509.2 | 432.4 | 270.6 | 232.7 | 229.1 | 233.5 | 2.6 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 9,498.7 | 1,785.9 | 1,718.5 | 1,584.5 | 1,479.1 | 21.2 | 2.7 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 8,902.3 | 1,354.0 | 395.2 | 340.5 | 337.1 | 1,106.3 | 3.4 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 37,071.4 | 3,789.1 | 3,723.9 | 3,520.0 | 3,381.3 | 24.3 | 4.7 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 32,745.7 | 2,270.2 | 1,080.3 | 1,002.5 | 978.5 | 1,148.5 | 7.5 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 76,032.6 | 13,214.5 | 12,756.7 | 11,833.4 | 10,739.2 | 64.0 | 5.6 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 59,288.4 | 3,844.5 | 1,895.2 | 1,845.5 | 1,805.7 | 2,425.9 | 17.1 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 67,185.4 | 23,355.8 | 21,644.8 | 19,989.5 | 17,423.9 | 257.6 | 12.6 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 58,625.9 | 4,085.9 | 1,938.9 | 1,878.0 | 1,853.6 | 9,982.3 | 61.9 | 3 | passed | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |

<a id="lfm2.5-230m-bf16-webgpu-default-ar"></a>
## lfm2.5-230m / akas / bf16 / webgpu / default / ar

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-230m-bf16-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / bf16 / webgpu / fp8 / ar

Profile status: **budget_exhausted**; qualification: **not_measured**; effective KV dtype: `fp8`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-230m-f16-cuda-default-ar"></a>
## lfm2.5-230m / akas / f16 / cuda / default / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json).

Peak device GiB: 7.6; peak host process RSS GiB: 2.3. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 8,687.9 | 481.7 | 478.8 | 437.7 | 427.0 | 4.6 | 2.3 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 10,022.4 | 437.2 | 258.6 | 223.7 | 220.9 | 255.6 | 2.6 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 10,775.0 | 1,074.3 | 1,056.1 | 994.4 | 954.7 | 20.2 | 4.1 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,873.7 | 909.6 | 320.0 | 301.5 | 297.5 | 1,120.2 | 4.9 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 41,971.6 | 4,035.0 | 3,970.6 | 3,717.9 | 3,556.4 | 23.4 | 4.4 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 30,588.9 | 2,375.9 | 1,062.4 | 1,005.7 | 992.9 | 1,170.9 | 7.1 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 71,411.7 | 13,294.8 | 12,799.1 | 11,953.2 | 11,250.3 | 52.4 | 5.5 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 57,428.8 | 3,518.4 | 1,783.8 | 1,742.4 | 1,709.8 | 2,490.0 | 18.6 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 66,044.6 | 21,848.4 | 20,326.2 | 19,904.0 | 17,701.2 | 359.5 | 12.6 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 56,229.9 | 3,617.0 | 1,789.4 | 1,751.1 | 1,729.8 | 10,381.5 | 69.1 | 3 | passed | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |

<a id="lfm2.5-230m-f16-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / f16 / cuda / fp8 / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json).

Peak device GiB: 4.4; peak host process RSS GiB: 2.3. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded; ValueError: model logits differ from qualified baseline: b1-p32

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 8,630.9 | 483.3 | 480.4 | 442.5 | 427.0 | 4.7 | 2.3 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 10,736.9 | 434.2 | 265.0 | 230.3 | 227.6 | 241.0 | 2.5 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 10,611.4 | 1,071.5 | 1,053.2 | 995.9 | 949.6 | 20.5 | 4.2 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,643.4 | 901.9 | 313.6 | 300.1 | 296.9 | 1,124.7 | 4.8 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 41,153.5 | 4,001.9 | 3,936.9 | 3,731.2 | 3,569.3 | 23.6 | 4.4 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 29,635.5 | 2,383.8 | 1,046.8 | 1,009.0 | 983.2 | 1,195.7 | 7.1 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 70,734.8 | 13,248.3 | 12,736.0 | 11,919.9 | 11,257.3 | 52.6 | 5.5 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 57,349.2 | 3,807.9 | 1,853.4 | 1,798.8 | 1,767.3 | 2,506.2 | 17.3 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 65,106.4 | 22,284.4 | 20,674.9 | 19,723.6 | 17,590.6 | 379.2 | 12.6 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 56,673.5 | 4,018.7 | 1,889.6 | 1,854.4 | 1,801.0 | 10,353.2 | 63.5 | 3 | passed | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |

<a id="lfm2.5-230m-f16-webgpu-default-ar"></a>
## lfm2.5-230m / akas / f16 / webgpu / default / ar

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-230m-f16-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / f16 / webgpu / fp8 / ar

Profile status: **budget_exhausted**; qualification: **not_measured**; effective KV dtype: `fp8`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-230m-q4_0-cuda-default-ar"></a>
## lfm2.5-230m / akas / q4_0 / cuda / default / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json).

Peak device GiB: 7.2; peak host process RSS GiB: 1.9. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded; ValueError: model logits differ from qualified baseline: b1-p2048

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 9,822.4 | 1,151.7 | 1,127.5 | 954.9 | 904.1 | 4.1 | 1.1 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 10,715.8 | 926.7 | 390.0 | 341.4 | 336.9 | 215.2 | 1.3 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 9,643.9 | 1,605.5 | 1,553.0 | 1,424.5 | 1,339.0 | 19.7 | 2.9 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,181.8 | 1,235.3 | 329.8 | 307.1 | 309.4 | 1,208.0 | 3.7 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 37,431.8 | 5,876.6 | 5,697.7 | 5,212.2 | 5,009.2 | 22.5 | 3.1 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 27,462.5 | 2,831.7 | 1,071.7 | 1,012.5 | 1,019.2 | 1,263.8 | 6.0 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 68,465.7 | 14,244.5 | 13,641.5 | 12,557.1 | 12,037.7 | 54.2 | 5.1 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 52,030.3 | 3,632.2 | 1,718.4 | 1,674.3 | 1,656.8 | 2,717.8 | 18.1 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 57,464.3 | 19,968.8 | 18,504.2 | 17,446.7 | 15,745.1 | 272.0 | 15.2 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 51,341.4 | 3,657.9 | 1,715.8 | 1,696.8 | 1,688.1 | 11,157.0 | 66.8 | 3 | passed | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |

<a id="lfm2.5-230m-q4_0-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / q4_0 / cuda / fp8 / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json).

Peak device GiB: 4.1; peak host process RSS GiB: 1.7. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded; ValueError: model logits differ from qualified baseline: b1-p32; ValueError: greedy output differs from baseline for B64/P2048

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 9,684.1 | 1,144.2 | 1,120.3 | 956.0 | 901.1 | 4.1 | 1.1 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 11,537.1 | 924.2 | 406.5 | 355.7 | 348.2 | 203.2 | 1.3 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 9,549.2 | 1,586.5 | 1,534.8 | 1,428.7 | 1,340.4 | 20.1 | 2.9 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 7,040.0 | 1,234.3 | 325.1 | 307.8 | 308.6 | 1,216.0 | 3.6 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 36,819.6 | 5,847.2 | 5,668.0 | 5,273.2 | 4,942.8 | 22.7 | 3.2 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 27,356.5 | 2,881.8 | 1,077.1 | 1,024.0 | 1,021.2 | 1,271.9 | 5.9 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 67,078.5 | 14,186.8 | 13,572.3 | 12,688.1 | 11,976.6 | 60.4 | 5.1 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 52,452.4 | 3,940.6 | 1,799.8 | 1,733.0 | 1,743.2 | 2,653.0 | 16.7 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 58,187.5 | 20,477.5 | 18,958.3 | 17,629.8 | 16,131.4 | 268.6 | 14.8 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 52,561.9 | 4,075.2 | 1,822.3 | 1,770.6 | 1,741.8 | 11,141.9 | 62.3 | 3 | passed | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |

<a id="lfm2.5-230m-q4_0-webgpu-default-ar"></a>
## lfm2.5-230m / akas / q4_0 / webgpu / default / ar

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-230m-q4_0-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / q4_0 / webgpu / fp8 / ar

Profile status: **budget_exhausted**; qualification: **not_measured**; effective KV dtype: `fp8`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-230m-q4_k_m-cuda-default-ar"></a>
## lfm2.5-230m / akas / q4_k_m / cuda / default / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json).

Peak device GiB: 7.3; peak host process RSS GiB: 1.8. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "9f34d665a4a48c1b", "model_options": {"kv_dtype": "float16", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 7,796.1 | 1,040.5 | 1,014.6 | 867.7 | 811.0 | 5.2 | 1.2 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 8,921.2 | 855.3 | 338.6 | 299.5 | 293.4 | 256.9 | 1.4 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 7,699.8 | 1,335.4 | 1,289.5 | 1,198.4 | 1,135.6 | 24.2 | 3.5 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 5,835.3 | 1,114.9 | 275.3 | 254.0 | 252.1 | 1,524.1 | 4.2 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 30,122.4 | 5,078.0 | 4,910.0 | 4,530.4 | 4,278.9 | 26.8 | 3.7 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 22,371.0 | 2,749.6 | 930.3 | 865.9 | 854.9 | 1,590.4 | 6.5 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 54,372.3 | 12,589.9 | 12,000.0 | 10,888.2 | 10,273.1 | 66.3 | 6.0 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 43,032.3 | 3,838.1 | 1,585.5 | 1,479.9 | 1,466.9 | 3,361.1 | 18.3 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 46,410.2 | 18,913.8 | 17,287.6 | 15,584.2 | 14,412.6 | 318.8 | 16.6 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 41,888.5 | 3,678.7 | 1,534.5 | 1,498.2 | 1,463.1 | 13,805.2 | 69.8 | 3 | passed | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |

<a id="lfm2.5-230m-q4_k_m-cuda-fp8-ar"></a>
## lfm2.5-230m / akas / q4_k_m / cuda / fp8 / ar

Profile status: **budget_exhausted**; qualification: **passed**; effective KV dtype: `fp8`. Evidence: [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json).

Peak device GiB: 4.1; peak host process RSS GiB: 1.8. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded; ValueError: model logits differ from qualified baseline: b1-p32

Selected recipe: `{"chunk": 32, "deferred": false, "graphs": true, "head": null, "id": "94654911ffcd8725", "model_options": {"kv_dtype": "fp8", "optimized": true}, "policy": {}, "prefill_requests": 32, "provider": "cuda", "target": "sm_89", "trials": 1, "window": null}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | native-token-step-v2 | 7,388.9 | 1,024.0 | 997.3 | 863.3 | 824.3 | 4.9 | 1.2 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 1 | 2048 | 128 | native-token-step-v2 | 9,430.7 | 845.5 | 348.3 | 308.8 | 303.6 | 242.6 | 1.4 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 4 | 32 | 128 | native-token-step-v2 | 7,525.6 | 1,323.4 | 1,277.2 | 1,201.3 | 1,120.6 | 24.2 | 3.5 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 4 | 2048 | 128 | native-token-step-v2 | 5,612.1 | 1,099.4 | 266.4 | 254.1 | 250.3 | 1,542.6 | 4.2 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 16 | 32 | 128 | native-token-step-v2 | 29,451.2 | 5,041.0 | 4,870.9 | 4,534.7 | 4,279.0 | 27.4 | 3.7 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 16 | 2048 | 128 | native-token-step-v2 | 21,711.0 | 2,683.0 | 902.5 | 875.4 | 861.7 | 1,591.3 | 6.3 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 64 | 32 | 128 | native-token-step-v2 | 53,312.2 | 12,274.5 | 11,692.8 | 10,938.7 | 10,280.7 | 66.7 | 6.0 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 64 | 2048 | 128 | native-token-step-v2 | 41,827.2 | 3,814.7 | 1,555.9 | 1,514.7 | 1,510.1 | 3,298.0 | 17.5 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 256 | 32 | 128 | native-token-step-v2 | 45,069.3 | 18,580.3 | 16,964.4 | 15,734.4 | 14,505.6 | 322.3 | 16.5 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| 256 | 2048 | 128 | native-token-step-v2 | 41,983.6 | 4,021.3 | 1,588.6 | 1,550.7 | 1,536.3 | 13,602.3 | 62.9 | 3 | passed | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |

<a id="lfm2.5-230m-q4_k_m-webgpu-default-ar"></a>
## lfm2.5-230m / akas / q4_k_m / webgpu / default / ar

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="lfm2.5-230m-q4_k_m-webgpu-fp8-ar"></a>
## lfm2.5-230m / akas / q4_k_m / webgpu / fp8 / ar

Profile status: **budget_exhausted**; qualification: **not_measured**; effective KV dtype: `fp8`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker wall-clock deadline exceeded

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="qwen3.5-0.8b-bf16-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / default / ar

Profile status: **budget_exhausted**; qualification: **not_measured**; effective KV dtype: `bfloat16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker exited with code -9; worker wall-clock deadline exceeded

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="qwen3.5-0.8b-bf16-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / default / mtp

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `bfloat16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="qwen3.5-0.8b-bf16-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-bf16-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / cuda / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-bf16-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / default / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-bf16-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / default / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-bf16-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-bf16-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / bf16 / webgpu / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-f16-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / f16 / cuda / default / ar

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `bfloat16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option; worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="qwen3.5-0.8b-f16-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / f16 / cuda / default / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Qwen GGUF MTP is not implemented; no supported speculative head for this checkpoint/backend

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-f16-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / f16 / cuda / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-f16-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / f16 / cuda / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-f16-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / default / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-f16-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / default / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-f16-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-f16-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / f16 / webgpu / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_0-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / default / ar

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `bfloat16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option; worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="qwen3.5-0.8b-q4_0-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / default / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Qwen GGUF MTP is not implemented; no supported speculative head for this checkpoint/backend

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_0-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_0-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / cuda / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_0-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / default / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_0-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / default / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_0-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_0-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_0 / webgpu / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_k_m-cuda-default-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / default / ar

Profile status: **failed**; qualification: **not_measured**; effective KV dtype: `bfloat16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option; worker exited with code -9

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

<a id="qwen3.5-0.8b-q4_k_m-cuda-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / default / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Qwen GGUF MTP is not implemented; no supported speculative head for this checkpoint/backend

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_k_m-cuda-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / cuda / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

dense native attention currently requires BF16 KV

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-default-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / default / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-default-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / default / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / fp8 / ar

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

<a id="qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp"></a>
## qwen3.5-0.8b / akas / q4_k_m / webgpu / fp8 / mtp

Profile status: **unsupported**; qualification: **unsupported**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

native BF16 kernels currently require CUDA

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 32 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | unsupported | — |

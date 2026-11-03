# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 uses concurrent token-arrival timestamps: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. native-gpu-prefill-ar127-v1 uses CUDA events around native prefill and autoregressive decode. Column labels distinguish native engine delivery, CUDA events and HTTP/SSE transports. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

<a id="lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar"></a>
## lfm2.5-24b-a2b-fp8 / akas / fp8 / cuda / float16 / ar

Profile status: **unqualified**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Retained measurements; no qualified deployment imported

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | native-gpu-prefill-ar127-v1 | 82,035.4 | 13,610.0 | — | — | — | — | — | 2 | unqualified | [20261102212132863812-0f22b4a3](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102212132863812-0f22b4a3/run.json) |

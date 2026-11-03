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
| 1 | 2048 | 128 | client-stream-burst-v1 | 56,820.3 | 293.8 | — | — | — | 36.0 | 3.4 | 3 | unqualified | [20261102230421074409-17a02c7a](../../runs/20261102230421074409-17a02c7a/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 78,350.1 | 820.9 | — | — | — | 104.6 | 4.9 | 3 | unqualified | [20261102230421074409-17a02c7a](../../runs/20261102230421074409-17a02c7a/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 83,856.9 | 1,941.5 | — | — | — | 390.8 | 8.2 | 3 | unqualified | [20261102230421074409-17a02c7a](../../runs/20261102230421074409-17a02c7a/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 81,538.1 | 5,638.1 | — | — | — | 1,607.5 | 11.4 | 3 | unqualified | [20261102230421074409-17a02c7a](../../runs/20261102230421074409-17a02c7a/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 80,929.3 | 13,327.7 | — | — | — | 6,478.3 | 19.2 | 3 | unqualified | [20261102230421074409-17a02c7a](../../runs/20261102230421074409-17a02c7a/run.json) |

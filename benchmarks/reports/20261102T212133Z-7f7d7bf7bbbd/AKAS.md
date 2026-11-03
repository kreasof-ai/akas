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
| 1 | 2048 | 128 | client-stream-burst-v1 | 57,719.4 | 301.2 | — | — | — | 35.5 | 3.3 | 3 | unqualified | [20261102212133108734-4a2dd18e](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102212133108734-4a2dd18e/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 80,733.9 | 836.8 | — | — | — | 101.5 | 4.8 | 3 | unqualified | [20261102212133108734-4a2dd18e](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102212133108734-4a2dd18e/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 84,352.1 | 2,025.2 | — | — | — | 388.5 | 7.9 | 3 | unqualified | [20261102212133108734-4a2dd18e](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102212133108734-4a2dd18e/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 82,476.7 | 5,875.3 | — | — | — | 1,589.2 | 10.9 | 3 | unqualified | [20261102212133108734-4a2dd18e](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102212133108734-4a2dd18e/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 82,193.5 | 12,290.5 | — | — | — | 6,378.7 | 20.8 | 3 | unqualified | [20261102212133108734-4a2dd18e](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102212133108734-4a2dd18e/run.json) |

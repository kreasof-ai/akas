# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

<a id="lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar"></a>
## lfm2.5-24b-a2b-fp8 / akas / fp8 / cuda / float16 / ar

Profile status: **unqualified**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Retained measurements; no qualified deployment imported

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 23,333.5 | 144.0 | — | — | — | 87.8 | 6.9 | 3 | unqualified | [20261101042910598214-f1012001](https://github.com/kreasof-ai/akas/blob/ada1484381a1042411453607cb9399d94c196398/benchmarks/runs/20261101042910598214-f1012001/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 26,384.5 | 304.5 | — | — | — | 310.5 | 13.1 | 3 | unqualified | [20261101042910598214-f1012001](https://github.com/kreasof-ai/akas/blob/ada1484381a1042411453607cb9399d94c196398/benchmarks/runs/20261101042910598214-f1012001/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 26,390.4 | 649.5 | — | — | — | 1,241.7 | 24.6 | 3 | unqualified | [20261101042910598214-f1012001](https://github.com/kreasof-ai/akas/blob/ada1484381a1042411453607cb9399d94c196398/benchmarks/runs/20261101042910598214-f1012001/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 26,356.8 | 1,528.2 | — | — | — | 4,973.0 | 41.9 | 3 | unqualified | [20261101042910598214-f1012001](https://github.com/kreasof-ai/akas/blob/ada1484381a1042411453607cb9399d94c196398/benchmarks/runs/20261101042910598214-f1012001/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 26,311.1 | 3,074.2 | — | — | — | 19,926.5 | 83.3 | 3 | unqualified | [20261101042910598214-f1012001](https://github.com/kreasof-ai/akas/blob/ada1484381a1042411453607cb9399d94c196398/benchmarks/runs/20261101042910598214-f1012001/run.json) |

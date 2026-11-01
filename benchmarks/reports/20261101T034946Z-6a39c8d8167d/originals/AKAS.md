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
| 1 | 2048 | 128 | client-stream-burst-v1 | 22,247.0 | 144.3 | — | — | — | 92.1 | 6.9 | 3 | unqualified | [20261101034946288520-60611d52](../../runs/20261101034946288520-60611d52/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 23,004.2 | 304.9 | — | — | — | 356.1 | 13.1 | 3 | unqualified | [20261101034946288520-60611d52](../../runs/20261101034946288520-60611d52/run.json) |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

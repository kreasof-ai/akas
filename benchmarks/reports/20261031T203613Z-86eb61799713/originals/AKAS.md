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
| 1 | 2048 | 128 | client-stream-burst-v1 | 18,629.8 | 115.6 | — | — | — | 109.9 | 8.7 | 3 | unqualified | [20261031203613246653-4a984cbd](../../runs/20261031203613246653-4a984cbd/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 18,647.8 | 204.8 | — | — | — | 439.3 | 19.5 | 3 | unqualified | [20261031203613246653-4a984cbd](../../runs/20261031203613246653-4a984cbd/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 18,611.6 | 560.3 | — | — | — | 1,760.6 | 28.6 | 3 | unqualified | [20261031203613246653-4a984cbd](../../runs/20261031203613246653-4a984cbd/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 18,608.2 | 1,336.1 | — | — | — | 7,043.8 | 47.9 | 3 | unqualified | [20261031203613246653-4a984cbd](../../runs/20261031203613246653-4a984cbd/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 18,553.7 | 2,694.8 | — | — | — | 28,257.8 | 95.0 | 3 | unqualified | [20261031203613246653-4a984cbd](../../runs/20261031203613246653-4a984cbd/run.json) |

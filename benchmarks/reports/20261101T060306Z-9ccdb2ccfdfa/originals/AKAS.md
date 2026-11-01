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
| 1 | 2048 | 128 | client-stream-burst-v1 | 23,458.9 | 143.9 | — | — | — | 87.3 | 6.9 | 3 | unqualified | [20261101060306871297-f51dac49](../../runs/20261101060306871297-f51dac49/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 27,310.9 | 300.4 | — | — | — | 300.0 | 13.3 | 3 | unqualified | [20261101060306871297-f51dac49](../../runs/20261101060306871297-f51dac49/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 27,301.9 | 650.0 | — | — | — | 1,200.2 | 24.6 | 3 | unqualified | [20261101060306871297-f51dac49](../../runs/20261101060306871297-f51dac49/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 27,231.2 | 1,529.3 | — | — | — | 4,813.3 | 41.8 | 3 | unqualified | [20261101060306871297-f51dac49](../../runs/20261101060306871297-f51dac49/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 27,113.5 | 3,056.4 | — | — | — | 19,336.8 | 83.8 | 3 | unqualified | [20261101060306871297-f51dac49](../../runs/20261101060306871297-f51dac49/run.json) |

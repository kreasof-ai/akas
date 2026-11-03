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
| 1 | 2048 | 128 | client-stream-burst-v1 | 57,842.3 | 300.6 | — | — | — | 35.4 | 3.3 | 3 | unqualified | [20261102184101326680-45d28851](../../runs/20261102184101326680-45d28851/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 76,155.3 | 838.6 | — | — | — | 107.6 | 4.8 | 3 | unqualified | [20261102184101326680-45d28851](../../runs/20261102184101326680-45d28851/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 76,405.3 | 2,102.8 | — | — | — | 428.9 | 7.6 | 3 | unqualified | [20261102184101326680-45d28851](../../runs/20261102184101326680-45d28851/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 75,947.8 | 5,930.2 | — | — | — | 1,725.8 | 10.8 | 3 | unqualified | [20261102184101326680-45d28851](../../runs/20261102184101326680-45d28851/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 75,185.6 | 11,867.2 | — | — | — | 6,973.3 | 21.6 | 3 | unqualified | [20261102184101326680-45d28851](../../runs/20261102184101326680-45d28851/run.json) |

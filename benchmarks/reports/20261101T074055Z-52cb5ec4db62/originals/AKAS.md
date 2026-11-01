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
| 1 | 2048 | 128 | client-stream-burst-v1 | 26,018.8 | 140.2 | — | — | — | 78.7 | 7.1 | 3 | unqualified | [20261101074055313788-b49b56d1](../../runs/20261101074055313788-b49b56d1/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 31,975.3 | 201.2 | — | — | — | 256.2 | 19.9 | 3 | unqualified | [20261101074055313788-b49b56d1](../../runs/20261101074055313788-b49b56d1/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 31,990.3 | 722.5 | — | — | — | 1,024.3 | 22.1 | 3 | unqualified | [20261101074055313788-b49b56d1](../../runs/20261101074055313788-b49b56d1/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 31,888.5 | 1,627.9 | — | — | — | 4,110.3 | 39.3 | 3 | unqualified | [20261101074055313788-b49b56d1](../../runs/20261101074055313788-b49b56d1/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 31,797.5 | 3,698.0 | — | — | — | 16,488.4 | 69.2 | 3 | unqualified | [20261101074055313788-b49b56d1](../../runs/20261101074055313788-b49b56d1/run.json) |

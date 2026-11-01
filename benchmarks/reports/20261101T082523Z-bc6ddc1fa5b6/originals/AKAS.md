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
| 1 | 2048 | 128 | client-stream-burst-v1 | 26,006.5 | 138.3 | — | — | — | 78.7 | 7.2 | 3 | unqualified | [20261101082523921626-bc85c457](../../runs/20261101082523921626-bc85c457/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 31,993.5 | 202.5 | — | — | — | 256.1 | 19.8 | 3 | unqualified | [20261101082523921626-bc85c457](../../runs/20261101082523921626-bc85c457/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 31,981.3 | 721.8 | — | — | — | 1,024.6 | 22.2 | 3 | unqualified | [20261101082523921626-bc85c457](../../runs/20261101082523921626-bc85c457/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 31,960.9 | 1,380.0 | — | — | — | 4,101.0 | 46.4 | 3 | unqualified | [20261101082523921626-bc85c457](../../runs/20261101082523921626-bc85c457/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 31,761.1 | 3,081.3 | — | — | — | 16,507.3 | 83.1 | 3 | unqualified | [20261101082523921626-bc85c457](../../runs/20261101082523921626-bc85c457/run.json) |

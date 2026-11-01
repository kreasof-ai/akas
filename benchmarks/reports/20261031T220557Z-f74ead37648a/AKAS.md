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
| 1 | 2048 | 128 | client-stream-burst-v1 | 20,534.6 | 130.5 | — | — | — | 99.7 | 7.7 | 3 | unqualified | [20261031220557911853-4cb13161](https://github.com/kreasof-ai/akas/blob/02323f759827d291e180ffa899ee21deb721d266/benchmarks/runs/20261031220557911853-4cb13161/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 21,206.9 | 203.5 | — | — | — | 386.3 | 19.7 | 3 | unqualified | [20261031220557911853-4cb13161](https://github.com/kreasof-ai/akas/blob/02323f759827d291e180ffa899ee21deb721d266/benchmarks/runs/20261031220557911853-4cb13161/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 21,186.1 | 588.3 | — | — | — | 1,546.7 | 27.2 | 3 | unqualified | [20261031220557911853-4cb13161](https://github.com/kreasof-ai/akas/blob/02323f759827d291e180ffa899ee21deb721d266/benchmarks/runs/20261031220557911853-4cb13161/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 21,130.8 | 1,428.0 | — | — | — | 6,202.9 | 44.8 | 3 | unqualified | [20261031220557911853-4cb13161](https://github.com/kreasof-ai/akas/blob/02323f759827d291e180ffa899ee21deb721d266/benchmarks/runs/20261031220557911853-4cb13161/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 21,143.9 | 2,824.6 | — | — | — | 24,796.2 | 90.6 | 3 | unqualified | [20261031220557911853-4cb13161](https://github.com/kreasof-ai/akas/blob/02323f759827d291e180ffa899ee21deb721d266/benchmarks/runs/20261031220557911853-4cb13161/run.json) |

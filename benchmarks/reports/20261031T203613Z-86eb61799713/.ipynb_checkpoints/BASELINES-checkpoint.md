# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

<a id="baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / vllm / fp8 / cuda / default / ar

Profile status: **invalidated**; qualification: **unqualified**; effective KV dtype: `bfloat16`. Evidence: [20261031180745769467-5ef944ee](../../runs/20261031180745769467-5ef944ee/run.json).

Peak device GiB: 38.4; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | request-phase-v1 | 28,078.9 | 159.7 | — | — | 147.3 | 72.9 | 6.3 | 3 | unqualified | [20261031180745769467-5ef944ee](../../runs/20261031180745769467-5ef944ee/run.json) |
| 4 | 2048 | 128 | request-phase-v1 | 27,166.0 | 353.1 | — | — | 324.3 | 301.6 | 11.3 | 3 | unqualified | [20261031180745769467-5ef944ee](../../runs/20261031180745769467-5ef944ee/run.json) |
| 16 | 2048 | 128 | request-phase-v1 | 28,207.8 | 597.3 | — | — | 539.1 | 1,161.7 | 26.8 | 3 | unqualified | [20261031180745769467-5ef944ee](../../runs/20261031180745769467-5ef944ee/run.json) |
| 64 | 2048 | 128 | request-phase-v1 | 27,502.4 | 1,168.8 | — | — | 927.2 | 4,765.8 | 54.8 | 3 | unqualified | [20261031180745769467-5ef944ee](../../runs/20261031180745769467-5ef944ee/run.json) |
| 256 | 2048 | 128 | request-phase-v1 | 23,298.7 | 2,886.6 | — | — | 1,208.2 | 22,502.9 | 88.7 | 3 | unqualified | [20261031180745769467-5ef944ee](../../runs/20261031180745769467-5ef944ee/run.json) |

<a id="baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / sglang / fp8 / cuda / default / ar

Profile status: **not_measured**; qualification: **not_measured**; effective KV dtype: `unrecorded`. Evidence: —.

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

No matching full-workload baseline before campaign start

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 4 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 16 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 64 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |
| 256 | 2048 | 128 | — | — | — | — | — | — | — | — | 0 | not_measured | — |

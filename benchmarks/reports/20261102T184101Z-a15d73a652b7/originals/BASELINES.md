# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Retained measurements; importing evidence does not qualify an AKAS deployment. All values are medians. Phase protocol labels describe the measured boundaries. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first. Unqualified values retain †. Original run records and source snapshots are unchanged.

<a id="baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / vllm / fp8 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261102182946501179-83424263](../../runs/20261102182946501179-83424263/run.json).

Peak device GiB: 119.3; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 27,790.0 | 344.0 | — | — | 288.8 | 73.7 | 2.9 | 3 | passed | [20261102182946501179-83424263](../../runs/20261102182946501179-83424263/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 25,222.3 | 773.2 | — | — | 623.2 | 324.8 | 5.2 | 3 | passed | [20261102182946501179-83424263](../../runs/20261102182946501179-83424263/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 25,663.6 | 1,161.5 | — | — | 1,028.9 | 1,276.8 | 13.8 | 3 | passed | [20261102182946501179-83424263](../../runs/20261102182946501179-83424263/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 25,963.6 | 1,508.8 | — | — | 1,358.6 | 5,048.3 | 42.4 | 3 | passed | [20261102182946501179-83424263](../../runs/20261102182946501179-83424263/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 24,477.2 | 3,198.1 | — | — | 1,452.2 | 21,419.5 | 80.0 | 3 | passed | [20261102182946501179-83424263](../../runs/20261102182946501179-83424263/run.json) |

<a id="baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / sglang / fp8 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261102183523780472-768136b6](../../runs/20261102183523780472-768136b6/run.json).

Peak device GiB: 120.3; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 37,178.2 | 343.7 | — | — | 301.1 | 55.1 | 2.9 | 3 | passed | [20261102183523780472-768136b6](../../runs/20261102183523780472-768136b6/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 39,472.7 | 893.1 | — | — | 760.4 | 207.5 | 4.5 | 3 | passed | [20261102183523780472-768136b6](../../runs/20261102183523780472-768136b6/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 39,649.1 | 1,385.4 | — | — | 1,300.5 | 826.4 | 11.5 | 3 | passed | [20261102183523780472-768136b6](../../runs/20261102183523780472-768136b6/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 39,721.6 | 1,917.4 | — | — | 1,879.7 | 3,299.8 | 33.4 | 3 | passed | [20261102183523780472-768136b6](../../runs/20261102183523780472-768136b6/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 39,188.9 | 2,197.6 | — | — | 2,182.7 | 13,378.5 | 116.5 | 3 | passed | [20261102183523780472-768136b6](../../runs/20261102183523780472-768136b6/run.json) |

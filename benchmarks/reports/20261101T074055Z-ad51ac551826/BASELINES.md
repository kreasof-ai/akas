# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Qualifying AKAS measurements verified against vLLM 0.31.0, SGLang 0.5.21, and FP64/precision-matched oracles in [20261101113717922004-dcb7262b](../runs/20261101113717922004-dcb7262b/run.json) (16/16 three-way greedy agreement on BOS and chat probes; matching FP8 Tensor Core logit noise floor across all three runtimes). All values are medians. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first.

<a id="baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / vllm / fp8 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261031183705001992-d62597ef](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031183705001992-d62597ef/run.json).

Peak device GiB: 38.5; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 28,016.3 | 159.7 | — | — | 147.4 | 73.1 | 6.3 | 3 | passed | [20261031183705001992-d62597ef](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031183705001992-d62597ef/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 27,059.3 | 357.0 | — | — | 324.5 | 302.7 | 11.2 | 3 | passed | [20261031183705001992-d62597ef](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031183705001992-d62597ef/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 28,078.3 | 600.2 | — | — | 539.3 | 1,167.0 | 26.7 | 3 | passed | [20261031183705001992-d62597ef](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031183705001992-d62597ef/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 27,312.1 | 1,169.9 | — | — | 925.7 | 4,799.0 | 54.7 | 3 | passed | [20261031183705001992-d62597ef](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031183705001992-d62597ef/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 23,224.5 | 2,890.9 | — | — | 1,208.9 | 22,574.7 | 88.6 | 3 | passed | [20261031183705001992-d62597ef](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031183705001992-d62597ef/run.json) |

<a id="baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / sglang / fp8 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261031182740133509-0f8da2dc](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031182740133509-0f8da2dc/run.json).

Peak device GiB: 38.9; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 25,915.5 | 146.7 | — | — | 135.4 | 79.0 | 6.8 | 3 | passed | [20261031182740133509-0f8da2dc](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031182740133509-0f8da2dc/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 27,521.3 | 316.6 | — | — | 294.3 | 297.7 | 12.6 | 3 | passed | [20261031182740133509-0f8da2dc](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031182740133509-0f8da2dc/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 28,033.2 | 550.6 | — | — | 533.9 | 1,168.9 | 29.1 | 3 | passed | [20261031182740133509-0f8da2dc](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031182740133509-0f8da2dc/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 28,369.9 | 905.4 | — | — | 896.9 | 4,620.1 | 70.7 | 3 | passed | [20261031182740133509-0f8da2dc](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031182740133509-0f8da2dc/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 28,358.3 | 1,266.4 | — | — | 1,267.8 | 18,488.0 | 202.1 | 3 | passed | [20261031182740133509-0f8da2dc](https://github.com/kreasof-ai/akas/blob/8cb24ec2cea506d26bdf3bfa902f3346ddbacd37/benchmarks/runs/20261031182740133509-0f8da2dc/run.json) |

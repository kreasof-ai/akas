# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Qualifying AKAS measurements verified against vLLM 0.31.0, SGLang 0.5.21, and FP64/precision-matched oracles in [20261101113717922004-dcb7262b](../runs/20261101113717922004-dcb7262b/run.json) (16/16 three-way greedy agreement on BOS and chat probes; matching FP8 Tensor Core logit noise floor across all three runtimes). All values are medians. client-stream-burst-v1 measures concurrent native-engine token delivery: TTFT includes queueing and prefill rate is total prompt tokens divided by maximum TTFT. Decode counts the remaining output tokens after the first.

<a id="lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar"></a>
## lfm2.5-24b-a2b-fp8 / akas / fp8 / cuda / float16 / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 2048, "decode_prefill_token_budget": 256, "graphs": true, "id": "retained-20261101074055313788-b49b56d1", "prefill_chunk": 2048, "prefill_requests": 256, "prefill_token_budget": 524288, "trials": 0}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 26,018.8 | 140.2 | — | — | — | 78.7 | 7.1 | 3 | passed | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 31,975.3 | 201.2 | — | — | — | 256.2 | 19.9 | 3 | passed | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 31,990.3 | 722.5 | — | — | — | 1,024.3 | 22.1 | 3 | passed | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 31,888.5 | 1,627.9 | — | — | — | 4,110.3 | 39.3 | 3 | passed | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 31,797.5 | 3,698.0 | — | — | — | 16,488.4 | 69.2 | 3 | passed | [20261101074055313788-b49b56d1](https://github.com/kreasof-ai/akas/blob/3c6a4bc78bb8bbc8b01bcb15b2403d2219d68e39/benchmarks/runs/20261101074055313788-b49b56d1/run.json) |

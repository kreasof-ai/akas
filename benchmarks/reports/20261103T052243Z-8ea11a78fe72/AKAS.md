# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Retained measurement timings with later correctness verification of the same program and checkpoint. No deployment search or tuning was performed during verification. Phase protocols, timing samples and frozen baseline records are unchanged. AKAS uses the native engine token stream; the frozen vendor runs use HTTP/SSE. All throughput values are medians over three repetitions; decode counts the 127 outputs after the first token. Missing values remain —. † retains an incomplete or failed qualification.

<a id="lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar"></a>
## lfm2.5-24b-a2b-fp8 / akas / fp8 / cuda / float16 / ar

Profile status: **verified measurement**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 2048, "decode_prefill_token_budget": 256, "graphs": true, "id": "verified-existing", "prefill_chunk": 2048, "prefill_requests": 16, "prefill_token_budget": 524288, "trials": 0}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 58,651.3 | 327.9 | — | — | — | 34.9 | 3.0 | 3 | passed | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 81,489.2 | 877.9 | — | — | — | 100.5 | 4.6 | 3 | passed | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 87,488.6 | 2,033.5 | — | — | — | 374.5 | 7.9 | 3 | passed | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 85,482.9 | 5,901.2 | — | — | — | 1,533.3 | 10.8 | 3 | passed | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 85,024.1 | 14,040.8 | — | — | — | 6,166.3 | 18.2 | 3 | passed | [20261103052243038756-378e9a45](https://github.com/kreasof-ai/akas/blob/9a0a9fe89bd88e078ca230a1d01ad65a0fa41a53/benchmarks/runs/20261103052243038756-378e9a45/run.json) |

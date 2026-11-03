# AKAS configurations

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

Retained measurement timings with later correctness verification of the same program and checkpoint. No deployment search or tuning was performed during verification. Phase protocols, timing samples and frozen baseline records are unchanged. AKAS uses the native engine token stream; the frozen vendor runs use HTTP/SSE. All throughput values are medians over three repetitions; decode counts the 127 outputs after the first token. Missing values remain —. † retains an incomplete or failed qualification.

<a id="lfm2.5-24b-a2b-fp8-fp8-cuda-float16-ar"></a>
## lfm2.5-24b-a2b-fp8 / akas / fp8 / cuda / float16 / ar

Profile status: **verified measurement**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261102230421074409-17a02c7a](https://github.com/kreasof-ai/akas/blob/e40a04579f272bde374eaf2b916e4cc219bc9236/benchmarks/runs/20261102230421074409-17a02c7a/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

Selected recipe: `{"chunk": 2048, "decode_prefill_token_budget": 256, "graphs": true, "id": "verified-existing", "prefill_chunk": 2048, "prefill_requests": 16, "prefill_token_budget": 524288, "trials": 0}`.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 56,820.3 | 293.8 | — | — | — | 36.0 | 3.4 | 3 | passed | [20261102230421074409-17a02c7a](https://github.com/kreasof-ai/akas/blob/e40a04579f272bde374eaf2b916e4cc219bc9236/benchmarks/runs/20261102230421074409-17a02c7a/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 78,350.1 | 820.9 | — | — | — | 104.6 | 4.9 | 3 | passed | [20261102230421074409-17a02c7a](https://github.com/kreasof-ai/akas/blob/e40a04579f272bde374eaf2b916e4cc219bc9236/benchmarks/runs/20261102230421074409-17a02c7a/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 83,856.9 | 1,941.5 | — | — | — | 390.8 | 8.2 | 3 | passed | [20261102230421074409-17a02c7a](https://github.com/kreasof-ai/akas/blob/e40a04579f272bde374eaf2b916e4cc219bc9236/benchmarks/runs/20261102230421074409-17a02c7a/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 81,538.1 | 5,638.1 | — | — | — | 1,607.5 | 11.4 | 3 | passed | [20261102230421074409-17a02c7a](https://github.com/kreasof-ai/akas/blob/e40a04579f272bde374eaf2b916e4cc219bc9236/benchmarks/runs/20261102230421074409-17a02c7a/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 80,929.3 | 13,327.7 | — | — | — | 6,478.3 | 19.2 | 3 | passed | [20261102230421074409-17a02c7a](https://github.com/kreasof-ai/akas/blob/e40a04579f272bde374eaf2b916e4cc219bc9236/benchmarks/runs/20261102230421074409-17a02c7a/run.json) |

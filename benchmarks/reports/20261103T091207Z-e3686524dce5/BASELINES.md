# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm2.5-24b-a2b-fp8-vllm-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / vllm / fp8 / cuda / default / ar

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `unrecorded`. Evidence: [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 25,618.1 | 333.8 | — | — | 277.0 | 79.9 | 3.0 | 3 | unqualified | [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 51,464.2 | 998.5 | — | — | 769.2 | 159.2 | 4.0 | 3 | unqualified | [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 78,728.2 | 2,107.2 | — | — | 1,759.5 | 416.2 | 7.6 | 3 | unqualified | [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 87,455.9 | 3,699.2 | — | — | 3,206.4 | 1,498.7 | 17.3 | 3 | unqualified | [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 86,126.2 | 5,071.3 | — | — | 4,301.0 | 6,087.4 | 50.5 | 3 | unqualified | [20261102190120772675-8245f839](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190120772675-8245f839/run.json) |

<a id="baseline-lfm2.5-24b-a2b-fp8-sglang-cuda-fp8-ar"></a>
## lfm2.5-24b-a2b-fp8 / sglang / fp8 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `unrecorded`. Evidence: [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json).

Peak device GiB: —; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 2048 | 128 | client-stream-burst-v1 | 34,099.5 | 250.8 | — | — | 223.3 | 60.1 | 4.0 | 3 | passed | [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json) |
| 4 | 2048 | 128 | client-stream-burst-v1 | 57,952.2 | 755.5 | — | — | 675.5 | 141.4 | 5.3 | 3 | passed | [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json) |
| 16 | 2048 | 128 | client-stream-burst-v1 | 92,650.8 | 1,705.6 | — | — | 1,606.2 | 353.7 | 9.4 | 3 | passed | [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json) |
| 64 | 2048 | 128 | client-stream-burst-v1 | 98,686.5 | 3,253.0 | — | — | 3,054.7 | 1,328.2 | 19.7 | 3 | passed | [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json) |
| 256 | 2048 | 128 | client-stream-burst-v1 | 100,233.6 | 4,479.9 | — | — | 4,219.8 | 5,230.7 | 57.1 | 3 | passed | [20261102190540048720-a1e7c623](https://github.com/kreasof-ai/akas/blob/4bf776253c90c575ad7759db1c06b7a85e403bc7/benchmarks/runs/20261102190540048720-a1e7c623/run.json) |

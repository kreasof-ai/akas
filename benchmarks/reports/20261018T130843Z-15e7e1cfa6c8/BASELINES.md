# Baseline profiles

[Campaign overview](README.md) · [Side-by-side tables](COMPARISON.md) · [CSV](measurements.csv)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

<a id="baseline-lfm2.5-230m-vllm-cuda-bf16-ar"></a>
## lfm2.5-230m / vllm / bf16 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json).

Peak device GiB: 19.2; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | request-phase-v1 | 2,889.6 | 454.2 | — | — | 439.9 | 11.1 | 2.2 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 1 | 2048 | 128 | request-phase-v1 | 115,655.1 | 441.4 | — | — | 418.0 | 17.7 | 2.3 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 4 | 32 | 128 | request-phase-v1 | 6,944.2 | 1,745.7 | — | — | 1,650.8 | 18.4 | 2.3 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 4 | 2048 | 128 | request-phase-v1 | 117,833.0 | 1,316.1 | — | — | 1,224.1 | 69.5 | 3.0 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 16 | 32 | 128 | request-phase-v1 | 22,006.7 | 6,792.7 | — | — | 6,330.7 | 23.3 | 2.4 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 16 | 2048 | 128 | request-phase-v1 | 106,718.4 | 2,748.2 | — | — | 1,421.1 | 307.1 | 5.8 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 64 | 32 | 128 | request-phase-v1 | 52,475.6 | 20,705.2 | — | — | 18,992.9 | 39.0 | 3.1 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 64 | 2048 | 128 | request-phase-v1 | 87,542.5 | 3,913.7 | — | — | 1,856.5 | 1,497.2 | 16.4 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 256 | 32 | 128 | request-phase-v1 | 91,362.6 | 37,002.7 | — | — | 34,137.9 | 89.7 | 6.9 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| 256 | 2048 | 128 | request-phase-v1 | 40,474.9 | 5,756.3 | — | — | 1,928.1 | 12,953.4 | 44.5 | 3 | passed | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |

<a id="baseline-lfm2.5-230m-sglang-cuda-bf16-ar"></a>
## lfm2.5-230m / sglang / bf16 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json).

Peak device GiB: 19.8; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | sglang-scheduler-phase-v1 | 2,050.4 | 449.0 | — | — | 426.6 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 1 | 2048 | 128 | sglang-scheduler-phase-v1 | 104,327.8 | 436.6 | — | — | 411.9 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 4 | 32 | 128 | sglang-scheduler-phase-v1 | 8,551.8 | 1,697.9 | — | — | 1,624.5 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 4 | 2048 | 128 | sglang-scheduler-phase-v1 | 105,143.8 | 1,247.8 | — | — | 1,165.7 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 16 | 32 | 128 | sglang-scheduler-phase-v1 | 27,579.2 | 6,462.8 | — | — | 6,126.0 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 16 | 2048 | 128 | sglang-scheduler-phase-v1 | 99,196.6 | 2,431.5 | — | — | 2,344.2 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 64 | 32 | 128 | sglang-scheduler-phase-v1 | 60,482.7 | 20,230.8 | — | — | 18,681.1 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 64 | 2048 | 128 | sglang-scheduler-phase-v1 | 97,806.0 | 3,177.8 | — | — | 3,125.9 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 256 | 32 | 128 | sglang-scheduler-phase-v1 | 67,298.5 | 41,130.2 | — | — | 37,332.0 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| 256 | 2048 | 128 | sglang-scheduler-phase-v1 | 96,963.0 | 3,410.1 | — | — | 3,397.3 | — | — | 3 | passed | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |

<a id="baseline-lfm2.5-230m-llama-cuda-f16-ar"></a>
## lfm2.5-230m / llama / f16 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json).

Peak device GiB: 7.5; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 5,315.6 | 407.5 | — | — | 397.6 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 63,531.5 | 387.9 | — | — | 352.9 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 16,604.0 | 1,240.4 | — | — | 1,193.3 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 63,447.3 | 979.9 | — | — | 771.1 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 30,378.5 | 2,506.0 | — | — | 2,091.5 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 174,600.6 | 1,316.4 | — | — | 999.0 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 30,160.2 | 3,385.7 | — | — | 1,642.5 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 465,662.9 | 1,630.5 | — | — | 661.6 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 25,135.7 | 3,677.5 | — | — | 1,393.9 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 1,215,219.0 | 2,988.6 | — | — | 467.4 | — | — | 3 | passed | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |

<a id="baseline-lfm2.5-230m-llama-cuda-q4_0-ar"></a>
## lfm2.5-230m / llama / q4_0 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json).

Peak device GiB: 7.2; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 7,574.0 | 752.7 | — | — | 728.8 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 63,568.9 | 687.4 | — | — | 580.8 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 18,140.6 | 2,146.8 | — | — | 2,007.2 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 63,711.8 | 1,528.1 | — | — | 1,073.2 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 32,940.9 | 3,856.0 | — | — | 2,961.8 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 192,085.2 | 1,660.4 | — | — | 1,184.6 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 27,146.3 | 4,770.9 | — | — | 1,880.8 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 524,760.3 | 1,903.2 | — | — | 720.9 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 29,625.8 | 5,320.0 | — | — | 1,580.9 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 1,414,231.1 | 3,493.7 | — | — | 496.0 | — | — | 3 | passed | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |

<a id="baseline-lfm2.5-230m-llama-cuda-q4_k_m-ar"></a>
## lfm2.5-230m / llama / q4_k_m / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json).

Peak device GiB: 7.2; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 6,632.1 | 727.1 | — | — | 704.9 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 60,120.4 | 663.5 | — | — | 559.8 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 16,684.0 | 2,064.3 | — | — | 1,923.0 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 59,963.7 | 1,477.8 | — | — | 1,034.3 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 31,236.7 | 3,737.8 | — | — | 2,881.1 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 183,798.7 | 1,604.4 | — | — | 1,171.3 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 24,070.6 | 4,597.6 | — | — | 1,863.7 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 505,065.2 | 1,838.3 | — | — | 713.4 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 27,065.3 | 5,157.2 | — | — | 1,555.4 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 1,372,941.7 | 3,374.9 | — | — | 486.9 | — | — | 3 | passed | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |

<a id="baseline-lfm2.5-230m-llama-vulkan-f16-ar"></a>
## lfm2.5-230m / llama / f16 / vulkan / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json).

Peak device GiB: 7.4; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 3,512.2 | 375.7 | — | — | 365.2 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 45,135.0 | 359.5 | — | — | 318.5 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 7,100.1 | 1,158.5 | — | — | 1,089.0 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 42,024.3 | 917.0 | — | — | 673.5 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 17,479.2 | 2,205.0 | — | — | 1,829.5 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 126,761.1 | 1,083.7 | — | — | 841.5 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 15,906.2 | 3,362.1 | — | — | 1,608.3 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 372,549.9 | 1,365.9 | — | — | 611.1 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 17,363.6 | 3,718.5 | — | — | 1,381.8 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 1,045,548.1 | 2,548.1 | — | — | 442.5 | — | — | 3 | passed | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |

<a id="baseline-lfm2.5-230m-llama-vulkan-q4_0-ar"></a>
## lfm2.5-230m / llama / q4_0 / vulkan / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json).

Peak device GiB: 7.1; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 4,018.6 | 678.3 | — | — | 641.6 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 47,682.2 | 637.9 | — | — | 522.3 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 9,405.5 | 1,706.2 | — | — | 1,576.6 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 48,778.2 | 1,259.5 | — | — | 876.7 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 21,445.0 | 2,515.5 | — | — | 2,076.2 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 142,506.7 | 1,207.7 | — | — | 922.5 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 20,201.6 | 3,658.8 | — | — | 1,686.8 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 407,718.1 | 1,498.9 | — | — | 649.0 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 22,380.7 | 3,873.4 | — | — | 1,419.5 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 1,118,218.7 | 2,739.7 | — | — | 460.0 | — | — | 3 | passed | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |

<a id="baseline-lfm2.5-230m-llama-vulkan-q4_k_m-ar"></a>
## lfm2.5-230m / llama / q4_k_m / vulkan / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `float16`. Evidence: [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json).

Peak device GiB: 7.0; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 3,465.8 | 648.6 | — | — | 617.9 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 43,773.8 | 615.4 | — | — | 499.1 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 8,171.1 | 1,539.1 | — | — | 1,414.7 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 42,753.5 | 1,146.8 | — | — | 789.7 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 18,705.9 | 2,167.3 | — | — | 1,845.5 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 126,078.3 | 1,083.3 | — | — | 838.1 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 17,399.1 | 3,554.4 | — | — | 1,660.4 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 373,141.7 | 1,389.6 | — | — | 625.0 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 19,717.9 | 3,836.3 | — | — | 1,411.1 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 1,067,970.3 | 2,602.1 | — | — | 451.0 | — | — | 3 | passed | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |

<a id="baseline-qwen3.5-0.8b-vllm-cuda-bf16-ar"></a>
## qwen3.5-0.8b / vllm / bf16 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json).

Peak device GiB: 19.2; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | request-phase-v1 | 924.1 | 146.9 | — | — | 142.4 | 34.6 | 6.8 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 1 | 2048 | 128 | request-phase-v1 | 31,334.5 | 141.3 | — | — | 132.8 | 65.4 | 7.1 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 4 | 32 | 128 | request-phase-v1 | 1,728.5 | 523.9 | — | — | 490.6 | 74.1 | 7.6 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 4 | 2048 | 128 | request-phase-v1 | 27,927.1 | 421.5 | — | — | 388.4 | 293.3 | 9.5 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 16 | 32 | 128 | request-phase-v1 | 6,154.3 | 1,656.7 | — | — | 1,563.2 | 83.2 | 9.7 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 16 | 2048 | 128 | request-phase-v1 | 27,720.3 | 862.6 | — | — | 790.4 | 1,182.1 | 18.5 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 64 | 32 | 128 | request-phase-v1 | 12,999.2 | 3,360.7 | — | — | 3,236.2 | 157.5 | 19.0 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 64 | 2048 | 128 | request-phase-v1 | 23,233.8 | 1,136.4 | — | — | 987.1 | 5,641.4 | 56.3 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 256 | 32 | 128 | request-phase-v1 | 13,822.8 | 4,466.2 | — | — | 4,333.2 | 592.6 | 57.3 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| 256 | 2048 | 128 | request-phase-v1 | 15,968.0 | 1,857.1 | — | — | 907.5 | 32,833.6 | 137.8 | 3 | passed | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |

<a id="baseline-qwen3.5-0.8b-vllm-cuda-bf16-mtp"></a>
## qwen3.5-0.8b / vllm / bf16 / cuda / default / mtp

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `bfloat16`. Evidence: [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json).

Peak device GiB: 20.4; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

B64/P32: greedy outputs changed between repetitions

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | request-phase-v1 | 823.9 | 200.5 | — | — | 190.3 | 38.8 | 5.0 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 1 | 2048 | 128 | request-phase-v1 | 31,068.1 | 190.7 | — | — | 174.8 | 65.9 | 5.2 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 4 | 32 | 128 | request-phase-v1 | 2,706.7 | 751.8 | — | — | 707.8 | 47.3 | 5.3 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 4 | 2048 | 128 | request-phase-v1 | 26,241.9 | 576.9 | — | — | 487.1 | 312.2 | 6.9 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 16 | 32 | 128 | request-phase-v1 | 5,684.6 | 1,867.5 | — | — | 1,737.8 | 90.1 | 8.6 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 16 | 2048 | 128 | request-phase-v1 | 25,624.5 | 986.1 | — | — | 799.3 | 1,278.8 | 16.2 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 64 | 32 | 128 | request-phase-v1 | 14,600.0 | 4,131.3 | — | — | 3,928.6 | 140.3 | 15.5 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 64 | 2048 | 128 | request-phase-v1 | 20,328.5 | 1,229.7 | — | — | 934.0 | 6,447.7 | 52.0 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 256 | 32 | 128 | request-phase-v1 | 11,781.8 | 5,279.1 | — | — | 5,120.4 | 695.3 | 48.5 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| 256 | 2048 | 128 | request-phase-v1 | 16,618.7 | 3,244.6 | — | — | 968.5 | 31,548.0 | 78.9 | 3 | unqualified | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |

<a id="baseline-qwen3.5-0.8b-sglang-cuda-bf16-ar"></a>
## qwen3.5-0.8b / sglang / bf16 / cuda / default / ar

Profile status: **passed**; qualification: **passed**; effective KV dtype: `bfloat16`. Evidence: [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json).

Peak device GiB: 20.5; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | sglang-scheduler-phase-v1 | 1,189.1 | 148.8 | — | — | 145.2 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 1 | 2048 | 128 | sglang-scheduler-phase-v1 | 29,066.9 | 144.2 | — | — | 134.4 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 4 | 32 | 128 | sglang-scheduler-phase-v1 | 4,312.4 | 540.5 | — | — | 527.3 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 4 | 2048 | 128 | sglang-scheduler-phase-v1 | 28,037.5 | 423.5 | — | — | 395.5 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 16 | 32 | 128 | sglang-scheduler-phase-v1 | 15,963.8 | 1,682.3 | — | — | 1,648.2 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 16 | 2048 | 128 | sglang-scheduler-phase-v1 | 28,738.6 | 826.2 | — | — | 801.5 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 64 | 32 | 128 | sglang-scheduler-phase-v1 | 21,238.8 | 3,369.1 | — | — | 3,263.9 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 64 | 2048 | 128 | sglang-scheduler-phase-v1 | 28,451.9 | 1,050.9 | — | — | 1,044.2 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 256 | 32 | 128 | sglang-scheduler-phase-v1 | 21,967.4 | 4,318.5 | — | — | 4,260.9 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| 256 | 2048 | 128 | sglang-scheduler-phase-v1 | 27,592.6 | 1,107.3 | — | — | 1,108.9 | — | — | 3 | passed | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |

<a id="baseline-qwen3.5-0.8b-sglang-cuda-bf16-mtp"></a>
## qwen3.5-0.8b / sglang / bf16 / cuda / default / mtp

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `bfloat16`. Evidence: [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json).

Peak device GiB: 20.8; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

ValueError: MTP output differs from the matching AR probe

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | sglang-scheduler-phase-v1 | 933.8 | 181.8 | — | — | 174.5 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 1 | 2048 | 128 | sglang-scheduler-phase-v1 | 25,544.1 | 187.1 | — | — | 168.4 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 4 | 32 | 128 | sglang-scheduler-phase-v1 | 3,567.8 | 625.0 | — | — | 602.6 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 4 | 2048 | 128 | sglang-scheduler-phase-v1 | 24,898.7 | 523.4 | — | — | 449.2 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 16 | 32 | 128 | sglang-scheduler-phase-v1 | 13,282.7 | 1,500.4 | — | — | 1,466.1 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 16 | 2048 | 128 | sglang-scheduler-phase-v1 | 26,068.6 | 844.9 | — | — | 746.2 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 64 | 32 | 128 | sglang-scheduler-phase-v1 | 19,014.3 | 2,935.2 | — | — | 2,843.0 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 64 | 2048 | 128 | sglang-scheduler-phase-v1 | 26,061.6 | 999.4 | — | — | 962.9 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 256 | 32 | 128 | sglang-scheduler-phase-v1 | 19,726.8 | 3,396.3 | — | — | 3,315.3 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| 256 | 2048 | 128 | sglang-scheduler-phase-v1 | 16,862.5 | 1,500.2 | — | — | 941.8 | — | — | 3 | unqualified | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |

<a id="baseline-qwen3.5-0.8b-llama-cuda-f16-ar"></a>
## qwen3.5-0.8b / llama / f16 / cuda / default / ar

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json).

Peak device GiB: 13.9; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

B64/P32: greedy outputs changed between repetitions; B256/P32: greedy outputs changed between repetitions

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 2,636.8 | 132.2 | — | — | 131.3 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 19,112.0 | 131.4 | — | — | 118.8 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 5,674.3 | 388.5 | — | — | 384.3 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 32,924.5 | 295.2 | — | — | 273.7 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 10,120.0 | 825.8 | — | — | 817.8 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 119,751.6 | 467.7 | — | — | 434.0 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 13,085.8 | 996.7 | — | — | 986.0 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 362,758.8 | 550.2 | — | — | 467.1 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 9,951.0 | 1,064.2 | — | — | 1,044.2 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 713,479.1 | 1,059.4 | — | — | 486.9 | — | — | 3 | unqualified | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |

<a id="baseline-qwen3.5-0.8b-llama-cuda-q4_0-ar"></a>
## qwen3.5-0.8b / llama / q4_0 / cuda / default / ar

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json).

Peak device GiB: 12.8; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

B4/P32: greedy outputs changed between repetitions; B16/P32: greedy outputs changed between repetitions; B16/P2048: greedy outputs changed between repetitions; B64/P32: greedy outputs changed between repetitions; B64/P2048: greedy outputs changed between repetitions; B256/P32: greedy outputs changed between repetitions; B256/P2048: greedy outputs changed between repetitions

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 3,739.6 | 263.5 | — | — | 260.1 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 18,641.9 | 252.4 | — | — | 207.8 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 7,192.6 | 671.0 | — | — | 659.7 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 33,187.6 | 416.3 | — | — | 374.3 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 10,485.1 | 1,015.8 | — | — | 1,004.9 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 119,770.5 | 512.8 | — | — | 477.5 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 11,174.0 | 1,068.1 | — | — | 1,057.1 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 369,641.7 | 571.1 | — | — | 490.4 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 10,332.6 | 1,108.1 | — | — | 1,088.3 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 743,784.9 | 1,104.0 | — | — | 512.7 | — | — | 3 | unqualified | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |

<a id="baseline-qwen3.5-0.8b-llama-cuda-q4_k_m-ar"></a>
## qwen3.5-0.8b / llama / q4_k_m / cuda / default / ar

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json).

Peak device GiB: 12.9; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

B4/P32: greedy outputs changed between repetitions; B16/P32: greedy outputs changed between repetitions; B16/P2048: greedy outputs changed between repetitions; B64/P32: greedy outputs changed between repetitions; B64/P2048: greedy outputs changed between repetitions; B256/P32: greedy outputs changed between repetitions; B256/P2048: greedy outputs changed between repetitions

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 3,597.1 | 251.1 | — | — | 247.8 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 17,700.3 | 241.2 | — | — | 198.5 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 6,882.8 | 630.4 | — | — | 620.2 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 32,045.2 | 395.8 | — | — | 357.3 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 7,549.1 | 984.2 | — | — | 974.5 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 115,418.9 | 496.9 | — | — | 463.1 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 11,353.2 | 1,074.8 | — | — | 1,063.0 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 359,008.9 | 574.0 | — | — | 491.7 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 9,865.4 | 1,094.3 | — | — | 1,074.5 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 725,525.6 | 1,071.8 | — | — | 500.0 | — | — | 3 | unqualified | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |

<a id="baseline-qwen3.5-0.8b-llama-vulkan-f16-ar"></a>
## qwen3.5-0.8b / llama / f16 / vulkan / default / ar

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json).

Peak device GiB: 13.7; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

B64/P32: greedy outputs changed between repetitions; B256/P32: greedy outputs changed between repetitions

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 2,117.4 | 119.6 | — | — | 118.5 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 11,683.0 | 116.7 | — | — | 101.1 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 3,942.2 | 330.1 | — | — | 325.5 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 21,510.9 | 240.2 | — | — | 218.6 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 6,213.1 | 567.7 | — | — | 563.5 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 76,799.4 | 315.0 | — | — | 293.2 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 4,241.4 | 737.5 | — | — | 732.0 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 245,588.9 | 386.4 | — | — | 331.6 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 7,280.7 | 777.6 | — | — | 764.7 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 496,155.0 | 738.9 | — | — | 341.8 | — | — | 3 | unqualified | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |

<a id="baseline-qwen3.5-0.8b-llama-vulkan-q4_0-ar"></a>
## qwen3.5-0.8b / llama / q4_0 / vulkan / default / ar

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json).

Peak device GiB: 12.7; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

B16/P32: greedy outputs changed between repetitions; B64/P32: greedy outputs changed between repetitions; B64/P2048: greedy outputs changed between repetitions; B256/P32: greedy outputs changed between repetitions; B256/P2048: greedy outputs changed between repetitions

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 2,577.9 | 219.7 | — | — | 216.2 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 12,151.3 | 213.3 | — | — | 167.3 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 4,493.9 | 479.9 | — | — | 468.0 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 22,545.4 | 314.4 | — | — | 282.1 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 6,636.4 | 635.7 | — | — | 631.0 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 80,657.9 | 340.5 | — | — | 317.3 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 8,828.4 | 772.6 | — | — | 764.8 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 252,305.6 | 399.1 | — | — | 343.8 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 7,447.6 | 784.9 | — | — | 772.2 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 509,747.0 | 757.8 | — | — | 352.1 | — | — | 3 | unqualified | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |

<a id="baseline-qwen3.5-0.8b-llama-vulkan-q4_k_m-ar"></a>
## qwen3.5-0.8b / llama / q4_k_m / vulkan / default / ar

Profile status: **failed**; qualification: **unqualified**; effective KV dtype: `float16`. Evidence: [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json).

Peak device GiB: 12.7; peak host process RSS GiB: —. Loading and in-process compilation are included in host RSS.

B16/P32: greedy outputs changed between repetitions; B64/P32: greedy outputs changed between repetitions; B256/P32: greedy outputs changed between repetitions

| B | Prefix | Output | Phase protocol | Prefill tok/s | Decode tok/s | Native output tok/s | Engine output tok/s | Burst output tok/s | TTFT ms | TPOT ms | Repeats | Qualification | Evidence |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 32 | 128 | llama-server-eval-v1 | 2,259.2 | 212.3 | — | — | 208.0 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 1 | 2048 | 128 | llama-server-eval-v1 | 11,317.8 | 207.7 | — | — | 160.6 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 4 | 32 | 128 | llama-server-eval-v1 | 4,299.8 | 448.7 | — | — | 441.7 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 4 | 2048 | 128 | llama-server-eval-v1 | 21,061.7 | 292.4 | — | — | 261.4 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 16 | 32 | 128 | llama-server-eval-v1 | 5,820.4 | 594.7 | — | — | 589.8 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 16 | 2048 | 128 | llama-server-eval-v1 | 75,003.0 | 318.8 | — | — | 297.6 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 64 | 32 | 128 | llama-server-eval-v1 | 8,675.7 | 763.1 | — | — | 755.8 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 64 | 2048 | 128 | llama-server-eval-v1 | 216,310.1 | 360.6 | — | — | 313.5 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 256 | 32 | 128 | llama-server-eval-v1 | 7,641.5 | 783.2 | — | — | 770.2 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |
| 256 | 2048 | 128 | llama-server-eval-v1 | 489,870.2 | 730.9 | — | — | 338.4 | — | — | 3 | unqualified | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |

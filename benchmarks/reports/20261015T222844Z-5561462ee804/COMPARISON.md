# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-230m / bf16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-lfm2.5-230m-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-lfm2.5-230m-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 9,330.4 | 9,298.8 | — | — | 2,889.6 | 2,050.4 |
| 1 | 2048 | 10,603.7 | 11,509.2 | — | — | 115,655.1 | 104,327.8 |
| 4 | 32 | 9,505.3 | 9,498.7 | — | — | 6,944.2 | 8,551.8 |
| 4 | 2048 | 8,852.7 | 8,902.3 | — | — | 117,833.0 | 105,143.8 |
| 16 | 32 | 37,055.6 | 37,071.4 | — | — | 22,006.7 | 27,579.2 |
| 16 | 2048 | 32,902.0 | 32,745.7 | — | — | 106,718.4 | 99,196.6 |
| 64 | 32 | 72,216.4 | 76,032.6 | — | — | 52,475.6 | 60,482.7 |
| 64 | 2048 | 58,701.7 | 59,288.4 | — | — | 87,542.5 | 97,806.0 |
| 256 | 32 | 66,450.6 | 67,185.4 | — | — | 91,362.6 | 67,298.5 |
| 256 | 2048 | 58,499.8 | 58,625.9 | — | — | 40,474.9 | 96,963.0 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-lfm2.5-230m-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-lfm2.5-230m-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 479.7 | 480.0 | — | — | 454.2 | 449.0 |
| 1 | 2048 | 431.3 | 432.4 | — | — | 441.4 | 436.6 |
| 4 | 32 | 1,784.2 | 1,785.9 | — | — | 1,745.7 | 1,697.9 |
| 4 | 2048 | 1,331.9 | 1,354.0 | — | — | 1,316.1 | 1,247.8 |
| 16 | 32 | 3,778.1 | 3,789.1 | — | — | 6,792.7 | 6,462.8 |
| 16 | 2048 | 2,233.3 | 2,270.2 | — | — | 2,748.2 | 2,431.5 |
| 64 | 32 | 13,129.8 | 13,214.5 | — | — | 20,705.2 | 20,230.8 |
| 64 | 2048 | 3,464.7 | 3,844.5 | — | — | 3,913.7 | 3,177.8 |
| 256 | 32 | 22,320.8 | 23,355.8 | — | — | 37,002.7 | 41,130.2 |
| 256 | 2048 | 3,686.0 | 4,085.9 | — | — | 5,756.3 | 3,410.1 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | [vllm cuda default / native call](BASELINES.md#baseline-lfm2.5-230m-vllm-cuda-bf16-ar) | [sglang cuda default / native call](BASELINES.md#baseline-lfm2.5-230m-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 424.1 | 425.0 | — | — | 439.9 | 426.6 |
| 1 | 2048 | 224.8 | 229.1 | — | — | 418.0 | 411.9 |
| 4 | 32 | 1,486.2 | 1,479.1 | — | — | 1,650.8 | 1,624.5 |
| 4 | 2048 | 344.1 | 337.1 | — | — | 1,224.1 | 1,165.7 |
| 16 | 32 | 3,405.8 | 3,381.3 | — | — | 6,330.7 | 6,126.0 |
| 16 | 2048 | 989.0 | 978.5 | — | — | 1,421.1 | 2,344.2 |
| 64 | 32 | 11,087.4 | 10,739.2 | — | — | 18,992.9 | 18,681.1 |
| 64 | 2048 | 1,722.4 | 1,805.7 | — | — | 1,856.5 | 3,125.9 |
| 256 | 32 | 17,372.1 | 17,423.9 | — | — | 34,137.9 | 37,332.0 |
| 256 | 2048 | 1,779.4 | 1,853.6 | — | — | 1,928.1 | 3,397.3 |

## lfm2.5-230m / f16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 8,687.9 | 8,630.9 | — | — | 5,315.6 | 3,512.2 |
| 1 | 2048 | 10,022.4 | 10,736.9 | — | — | 63,531.5 | 45,135.0 |
| 4 | 32 | 10,775.0 | 10,611.4 | — | — | 16,604.0 | 7,100.1 |
| 4 | 2048 | 7,873.7 | 7,643.4 | — | — | 63,447.3 | 42,024.3 |
| 16 | 32 | 41,971.6 | 41,153.5 | — | — | 30,378.5 | 17,479.2 |
| 16 | 2048 | 30,588.9 | 29,635.5 | — | — | 174,600.6 | 126,761.1 |
| 64 | 32 | 71,411.7 | 70,734.8 | — | — | 30,160.2 | 15,906.2 |
| 64 | 2048 | 57,428.8 | 57,349.2 | — | — | 465,662.9 | 372,549.9 |
| 256 | 32 | 66,044.6 | 65,106.4 | — | — | 25,135.7 | 17,363.6 |
| 256 | 2048 | 56,229.9 | 56,673.5 | — | — | 1,215,219.0 | 1,045,548.1 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 481.7 | 483.3 | — | — | 407.5 | 375.7 |
| 1 | 2048 | 437.2 | 434.2 | — | — | 387.9 | 359.5 |
| 4 | 32 | 1,074.3 | 1,071.5 | — | — | 1,240.4 | 1,158.5 |
| 4 | 2048 | 909.6 | 901.9 | — | — | 979.9 | 917.0 |
| 16 | 32 | 4,035.0 | 4,001.9 | — | — | 2,506.0 | 2,205.0 |
| 16 | 2048 | 2,375.9 | 2,383.8 | — | — | 1,316.4 | 1,083.7 |
| 64 | 32 | 13,294.8 | 13,248.3 | — | — | 3,385.7 | 3,362.1 |
| 64 | 2048 | 3,518.4 | 3,807.9 | — | — | 1,630.5 | 1,365.9 |
| 256 | 32 | 21,848.4 | 22,284.4 | — | — | 3,677.5 | 3,718.5 |
| 256 | 2048 | 3,617.0 | 4,018.7 | — | — | 2,988.6 | 2,548.1 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-f16-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 427.0 | 427.0 | — | — | 397.6 | 365.2 |
| 1 | 2048 | 220.9 | 227.6 | — | — | 352.9 | 318.5 |
| 4 | 32 | 954.7 | 949.6 | — | — | 1,193.3 | 1,089.0 |
| 4 | 2048 | 297.5 | 296.9 | — | — | 771.1 | 673.5 |
| 16 | 32 | 3,556.4 | 3,569.3 | — | — | 2,091.5 | 1,829.5 |
| 16 | 2048 | 992.9 | 983.2 | — | — | 999.0 | 841.5 |
| 64 | 32 | 11,250.3 | 11,257.3 | — | — | 1,642.5 | 1,608.3 |
| 64 | 2048 | 1,709.8 | 1,767.3 | — | — | 661.6 | 611.1 |
| 256 | 32 | 17,701.2 | 17,590.6 | — | — | 1,393.9 | 1,381.8 |
| 256 | 2048 | 1,729.8 | 1,801.0 | — | — | 467.4 | 442.5 |

## lfm2.5-230m / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 9,822.4 | 9,684.1 | — | — | 7,574.0 | 4,018.6 |
| 1 | 2048 | 10,715.8 | 11,537.1 | — | — | 63,568.9 | 47,682.2 |
| 4 | 32 | 9,643.9 | 9,549.2 | — | — | 18,140.6 | 9,405.5 |
| 4 | 2048 | 7,181.8 | 7,040.0 | — | — | 63,711.8 | 48,778.2 |
| 16 | 32 | 37,431.8 | 36,819.6 | — | — | 32,940.9 | 21,445.0 |
| 16 | 2048 | 27,462.5 | 27,356.5 | — | — | 192,085.2 | 142,506.7 |
| 64 | 32 | 68,465.7 | 67,078.5 | — | — | 27,146.3 | 20,201.6 |
| 64 | 2048 | 52,030.3 | 52,452.4 | — | — | 524,760.3 | 407,718.1 |
| 256 | 32 | 57,464.3 | 58,187.5 | — | — | 29,625.8 | 22,380.7 |
| 256 | 2048 | 51,341.4 | 52,561.9 | — | — | 1,414,231.1 | 1,118,218.7 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 1,151.7 | 1,144.2 | — | — | 752.7 | 678.3 |
| 1 | 2048 | 926.7 | 924.2 | — | — | 687.4 | 637.9 |
| 4 | 32 | 1,605.5 | 1,586.5 | — | — | 2,146.8 | 1,706.2 |
| 4 | 2048 | 1,235.3 | 1,234.3 | — | — | 1,528.1 | 1,259.5 |
| 16 | 32 | 5,876.6 | 5,847.2 | — | — | 3,856.0 | 2,515.5 |
| 16 | 2048 | 2,831.7 | 2,881.8 | — | — | 1,660.4 | 1,207.7 |
| 64 | 32 | 14,244.5 | 14,186.8 | — | — | 4,770.9 | 3,658.8 |
| 64 | 2048 | 3,632.2 | 3,940.6 | — | — | 1,903.2 | 1,498.9 |
| 256 | 32 | 19,968.8 | 20,477.5 | — | — | 5,320.0 | 3,873.4 |
| 256 | 2048 | 3,657.9 | 4,075.2 | — | — | 3,493.7 | 2,739.7 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_0-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 904.1 | 901.1 | — | — | 728.8 | 641.6 |
| 1 | 2048 | 336.9 | 348.2 | — | — | 580.8 | 522.3 |
| 4 | 32 | 1,339.0 | 1,340.4 | — | — | 2,007.2 | 1,576.6 |
| 4 | 2048 | 309.4 | 308.6 | — | — | 1,073.2 | 876.7 |
| 16 | 32 | 5,009.2 | 4,942.8 | — | — | 2,961.8 | 2,076.2 |
| 16 | 2048 | 1,019.2 | 1,021.2 | — | — | 1,184.6 | 922.5 |
| 64 | 32 | 12,037.7 | 11,976.6 | — | — | 1,880.8 | 1,686.8 |
| 64 | 2048 | 1,656.8 | 1,743.2 | — | — | 720.9 | 649.0 |
| 256 | 32 | 15,745.1 | 16,131.4 | — | — | 1,580.9 | 1,419.5 |
| 256 | 2048 | 1,688.1 | 1,741.8 | — | — | 496.0 | 460.0 |

## lfm2.5-230m / q4_k_m / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 7,796.1 | 7,388.9 | — | — | 6,632.1 | 3,465.8 |
| 1 | 2048 | 8,921.2 | 9,430.7 | — | — | 60,120.4 | 43,773.8 |
| 4 | 32 | 7,699.8 | 7,525.6 | — | — | 16,684.0 | 8,171.1 |
| 4 | 2048 | 5,835.3 | 5,612.1 | — | — | 59,963.7 | 42,753.5 |
| 16 | 32 | 30,122.4 | 29,451.2 | — | — | 31,236.7 | 18,705.9 |
| 16 | 2048 | 22,371.0 | 21,711.0 | — | — | 183,798.7 | 126,078.3 |
| 64 | 32 | 54,372.3 | 53,312.2 | — | — | 24,070.6 | 17,399.1 |
| 64 | 2048 | 43,032.3 | 41,827.2 | — | — | 505,065.2 | 373,141.7 |
| 256 | 32 | 46,410.2 | 45,069.3 | — | — | 27,065.3 | 19,717.9 |
| 256 | 2048 | 41,888.5 | 41,983.6 | — | — | 1,372,941.7 | 1,067,970.3 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 1,040.5 | 1,024.0 | — | — | 727.1 | 648.6 |
| 1 | 2048 | 855.3 | 845.5 | — | — | 663.5 | 615.4 |
| 4 | 32 | 1,335.4 | 1,323.4 | — | — | 2,064.3 | 1,539.1 |
| 4 | 2048 | 1,114.9 | 1,099.4 | — | — | 1,477.8 | 1,146.8 |
| 16 | 32 | 5,078.0 | 5,041.0 | — | — | 3,737.8 | 2,167.3 |
| 16 | 2048 | 2,749.6 | 2,683.0 | — | — | 1,604.4 | 1,083.3 |
| 64 | 32 | 12,589.9 | 12,274.5 | — | — | 4,597.6 | 3,554.4 |
| 64 | 2048 | 3,838.1 | 3,814.7 | — | — | 1,838.3 | 1,389.6 |
| 256 | 32 | 18,913.8 | 18,580.3 | — | — | 5,157.2 | 3,836.3 |
| 256 | 2048 | 3,678.7 | 4,021.3 | — | — | 3,374.9 | 2,602.1 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_k_m-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 811.0 | 824.3 | — | — | 704.9 | 617.9 |
| 1 | 2048 | 293.4 | 303.6 | — | — | 559.8 | 499.1 |
| 4 | 32 | 1,135.6 | 1,120.6 | — | — | 1,923.0 | 1,414.7 |
| 4 | 2048 | 252.1 | 250.3 | — | — | 1,034.3 | 789.7 |
| 16 | 32 | 4,278.9 | 4,279.0 | — | — | 2,881.1 | 1,845.5 |
| 16 | 2048 | 854.9 | 861.7 | — | — | 1,171.3 | 838.1 |
| 64 | 32 | 10,273.1 | 10,280.7 | — | — | 1,863.7 | 1,660.4 |
| 64 | 2048 | 1,466.9 | 1,510.1 | — | — | 713.4 | 625.0 |
| 256 | 32 | 14,412.6 | 14,505.6 | — | — | 1,555.4 | 1,411.1 |
| 256 | 2048 | 1,463.1 | 1,536.3 | — | — | 486.9 | 451.0 |

## qwen3.5-0.8b / bf16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 924.1 | 1,189.1 |
| 1 | 2048 | — | — | — | — | 31,334.5 | 29,066.9 |
| 4 | 32 | — | — | — | — | 1,728.5 | 4,312.4 |
| 4 | 2048 | — | — | — | — | 27,927.1 | 28,037.5 |
| 16 | 32 | — | — | — | — | 6,154.3 | 15,963.8 |
| 16 | 2048 | — | — | — | — | 27,720.3 | 28,738.6 |
| 64 | 32 | — | — | — | — | 12,999.2 | 21,238.8 |
| 64 | 2048 | — | — | — | — | 23,233.8 | 28,451.9 |
| 256 | 32 | — | — | — | — | 13,822.8 | 21,967.4 |
| 256 | 2048 | — | — | — | — | 15,968.0 | 27,592.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 146.9 | 148.8 |
| 1 | 2048 | — | — | — | — | 141.3 | 144.2 |
| 4 | 32 | — | — | — | — | 523.9 | 540.5 |
| 4 | 2048 | — | — | — | — | 421.5 | 423.5 |
| 16 | 32 | — | — | — | — | 1,656.7 | 1,682.3 |
| 16 | 2048 | — | — | — | — | 862.6 | 826.2 |
| 64 | 32 | — | — | — | — | 3,360.7 | 3,369.1 |
| 64 | 2048 | — | — | — | — | 1,136.4 | 1,050.9 |
| 256 | 32 | — | — | — | — | 4,466.2 | 4,318.5 |
| 256 | 2048 | — | — | — | — | 1,857.1 | 1,107.3 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | [vllm cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-ar) | [sglang cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 142.4 | 145.2 |
| 1 | 2048 | — | — | — | — | 132.8 | 134.4 |
| 4 | 32 | — | — | — | — | 490.6 | 527.3 |
| 4 | 2048 | — | — | — | — | 388.4 | 395.5 |
| 16 | 32 | — | — | — | — | 1,563.2 | 1,648.2 |
| 16 | 2048 | — | — | — | — | 790.4 | 801.5 |
| 64 | 32 | — | — | — | — | 3,236.2 | 3,263.9 |
| 64 | 2048 | — | — | — | — | 987.1 | 1,044.2 |
| 256 | 32 | — | — | — | — | 4,333.2 | 4,260.9 |
| 256 | 2048 | — | — | — | — | 907.5 | 1,108.9 |

## qwen3.5-0.8b / bf16 / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-mtp) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 823.9 † | 933.8 † |
| 1 | 2048 | — | — | — | — | 31,068.1 † | 25,544.1 † |
| 4 | 32 | — | — | — | — | 2,706.7 † | 3,567.8 † |
| 4 | 2048 | — | — | — | — | 26,241.9 † | 24,898.7 † |
| 16 | 32 | — | — | — | — | 5,684.6 † | 13,282.7 † |
| 16 | 2048 | — | — | — | — | 25,624.5 † | 26,068.6 † |
| 64 | 32 | — | — | — | — | 14,600.0 † | 19,014.3 † |
| 64 | 2048 | — | — | — | — | 20,328.5 † | 26,061.6 † |
| 256 | 32 | — | — | — | — | 11,781.8 † | 19,726.8 † |
| 256 | 2048 | — | — | — | — | 16,618.7 † | 16,862.5 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-mtp) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 200.5 † | 181.8 † |
| 1 | 2048 | — | — | — | — | 190.7 † | 187.1 † |
| 4 | 32 | — | — | — | — | 751.8 † | 625.0 † |
| 4 | 2048 | — | — | — | — | 576.9 † | 523.4 † |
| 16 | 32 | — | — | — | — | 1,867.5 † | 1,500.4 † |
| 16 | 2048 | — | — | — | — | 986.1 † | 844.9 † |
| 64 | 32 | — | — | — | — | 4,131.3 † | 2,935.2 † |
| 64 | 2048 | — | — | — | — | 1,229.7 † | 999.4 † |
| 256 | 32 | — | — | — | — | 5,279.1 † | 3,396.3 † |
| 256 | 2048 | — | — | — | — | 3,244.6 † | 1,500.2 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | [vllm cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-mtp) | [sglang cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 190.3 † | 174.5 † |
| 1 | 2048 | — | — | — | — | 174.8 † | 168.4 † |
| 4 | 32 | — | — | — | — | 707.8 † | 602.6 † |
| 4 | 2048 | — | — | — | — | 487.1 † | 449.2 † |
| 16 | 32 | — | — | — | — | 1,737.8 † | 1,466.1 † |
| 16 | 2048 | — | — | — | — | 799.3 † | 746.2 † |
| 64 | 32 | — | — | — | — | 3,928.6 † | 2,843.0 † |
| 64 | 2048 | — | — | — | — | 934.0 † | 962.9 † |
| 256 | 32 | — | — | — | — | 5,120.4 † | 3,315.3 † |
| 256 | 2048 | — | — | — | — | 968.5 † | 941.8 † |

## qwen3.5-0.8b / f16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 2,636.8 † | 2,117.4 † |
| 1 | 2048 | — | — | — | — | 19,112.0 † | 11,683.0 † |
| 4 | 32 | — | — | — | — | 5,674.3 † | 3,942.2 † |
| 4 | 2048 | — | — | — | — | 32,924.5 † | 21,510.9 † |
| 16 | 32 | — | — | — | — | 10,120.0 † | 6,213.1 † |
| 16 | 2048 | — | — | — | — | 119,751.6 † | 76,799.4 † |
| 64 | 32 | — | — | — | — | 13,085.8 † | 4,241.4 † |
| 64 | 2048 | — | — | — | — | 362,758.8 † | 245,588.9 † |
| 256 | 32 | — | — | — | — | 9,951.0 † | 7,280.7 † |
| 256 | 2048 | — | — | — | — | 713,479.1 † | 496,155.0 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 132.2 † | 119.6 † |
| 1 | 2048 | — | — | — | — | 131.4 † | 116.7 † |
| 4 | 32 | — | — | — | — | 388.5 † | 330.1 † |
| 4 | 2048 | — | — | — | — | 295.2 † | 240.2 † |
| 16 | 32 | — | — | — | — | 825.8 † | 567.7 † |
| 16 | 2048 | — | — | — | — | 467.7 † | 315.0 † |
| 64 | 32 | — | — | — | — | 996.7 † | 737.5 † |
| 64 | 2048 | — | — | — | — | 550.2 † | 386.4 † |
| 256 | 32 | — | — | — | — | 1,064.2 † | 777.6 † |
| 256 | 2048 | — | — | — | — | 1,059.4 † | 738.9 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-f16-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 131.3 † | 118.5 † |
| 1 | 2048 | — | — | — | — | 118.8 † | 101.1 † |
| 4 | 32 | — | — | — | — | 384.3 † | 325.5 † |
| 4 | 2048 | — | — | — | — | 273.7 † | 218.6 † |
| 16 | 32 | — | — | — | — | 817.8 † | 563.5 † |
| 16 | 2048 | — | — | — | — | 434.0 † | 293.2 † |
| 64 | 32 | — | — | — | — | 986.0 † | 732.0 † |
| 64 | 2048 | — | — | — | — | 467.1 † | 331.6 † |
| 256 | 32 | — | — | — | — | 1,044.2 † | 764.7 † |
| 256 | 2048 | — | — | — | — | 486.9 † | 341.8 † |

## qwen3.5-0.8b / f16 / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

## qwen3.5-0.8b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 3,739.6 † | 2,577.9 † |
| 1 | 2048 | — | — | — | — | 18,641.9 † | 12,151.3 † |
| 4 | 32 | — | — | — | — | 7,192.6 † | 4,493.9 † |
| 4 | 2048 | — | — | — | — | 33,187.6 † | 22,545.4 † |
| 16 | 32 | — | — | — | — | 10,485.1 † | 6,636.4 † |
| 16 | 2048 | — | — | — | — | 119,770.5 † | 80,657.9 † |
| 64 | 32 | — | — | — | — | 11,174.0 † | 8,828.4 † |
| 64 | 2048 | — | — | — | — | 369,641.7 † | 252,305.6 † |
| 256 | 32 | — | — | — | — | 10,332.6 † | 7,447.6 † |
| 256 | 2048 | — | — | — | — | 743,784.9 † | 509,747.0 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 263.5 † | 219.7 † |
| 1 | 2048 | — | — | — | — | 252.4 † | 213.3 † |
| 4 | 32 | — | — | — | — | 671.0 † | 479.9 † |
| 4 | 2048 | — | — | — | — | 416.3 † | 314.4 † |
| 16 | 32 | — | — | — | — | 1,015.8 † | 635.7 † |
| 16 | 2048 | — | — | — | — | 512.8 † | 340.5 † |
| 64 | 32 | — | — | — | — | 1,068.1 † | 772.6 † |
| 64 | 2048 | — | — | — | — | 571.1 † | 399.1 † |
| 256 | 32 | — | — | — | — | 1,108.1 † | 784.9 † |
| 256 | 2048 | — | — | — | — | 1,104.0 † | 757.8 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_0-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 260.1 † | 216.2 † |
| 1 | 2048 | — | — | — | — | 207.8 † | 167.3 † |
| 4 | 32 | — | — | — | — | 659.7 † | 468.0 † |
| 4 | 2048 | — | — | — | — | 374.3 † | 282.1 † |
| 16 | 32 | — | — | — | — | 1,004.9 † | 631.0 † |
| 16 | 2048 | — | — | — | — | 477.5 † | 317.3 † |
| 64 | 32 | — | — | — | — | 1,057.1 † | 764.8 † |
| 64 | 2048 | — | — | — | — | 490.4 † | 343.8 † |
| 256 | 32 | — | — | — | — | 1,088.3 † | 772.2 † |
| 256 | 2048 | — | — | — | — | 512.7 † | 352.1 † |

## qwen3.5-0.8b / q4_0 / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

## qwen3.5-0.8b / q4_k_m / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 3,597.1 † | 2,259.2 † |
| 1 | 2048 | — | — | — | — | 17,700.3 † | 11,317.8 † |
| 4 | 32 | — | — | — | — | 6,882.8 † | 4,299.8 † |
| 4 | 2048 | — | — | — | — | 32,045.2 † | 21,061.7 † |
| 16 | 32 | — | — | — | — | 7,549.1 † | 5,820.4 † |
| 16 | 2048 | — | — | — | — | 115,418.9 † | 75,003.0 † |
| 64 | 32 | — | — | — | — | 11,353.2 † | 8,675.7 † |
| 64 | 2048 | — | — | — | — | 359,008.9 † | 216,310.1 † |
| 256 | 32 | — | — | — | — | 9,865.4 † | 7,641.5 † |
| 256 | 2048 | — | — | — | — | 725,525.6 † | 489,870.2 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 251.1 † | 212.3 † |
| 1 | 2048 | — | — | — | — | 241.2 † | 207.7 † |
| 4 | 32 | — | — | — | — | 630.4 † | 448.7 † |
| 4 | 2048 | — | — | — | — | 395.8 † | 292.4 † |
| 16 | 32 | — | — | — | — | 984.2 † | 594.7 † |
| 16 | 2048 | — | — | — | — | 496.9 † | 318.8 † |
| 64 | 32 | — | — | — | — | 1,074.8 † | 763.1 † |
| 64 | 2048 | — | — | — | — | 574.0 † | 360.6 † |
| 256 | 32 | — | — | — | — | 1,094.3 † | 783.2 † |
| 256 | 2048 | — | — | — | — | 1,071.8 † | 730.9 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_k_m-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — | 247.8 † | 208.0 † |
| 1 | 2048 | — | — | — | — | 198.5 † | 160.6 † |
| 4 | 32 | — | — | — | — | 620.2 † | 441.7 † |
| 4 | 2048 | — | — | — | — | 357.3 † | 261.4 † |
| 16 | 32 | — | — | — | — | 974.5 † | 589.8 † |
| 16 | 2048 | — | — | — | — | 463.1 † | 297.6 † |
| 64 | 32 | — | — | — | — | 1,063.0 † | 755.8 † |
| 64 | 2048 | — | — | — | — | 491.7 † | 313.5 † |
| 256 | 32 | — | — | — | — | 1,074.5 † | 770.2 † |
| 256 | 2048 | — | — | — | — | 500.0 † | 338.4 † |

## qwen3.5-0.8b / q4_k_m / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | [akas cuda fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | [akas webgpu default / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | [akas webgpu fp8 / unmeasured](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | — | — | — | — |
| 1 | 2048 | — | — | — | — |
| 4 | 32 | — | — | — | — |
| 4 | 2048 | — | — | — | — |
| 16 | 32 | — | — | — | — |
| 16 | 2048 | — | — | — | — |
| 64 | 32 | — | — | — | — |
| 64 | 2048 | — | — | — | — |
| 256 | 32 | — | — | — | — |
| 256 | 2048 | — | — | — | — |

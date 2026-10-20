# Side-by-side throughput

[Campaign overview](README.md) · [Native prefill/decode and latency: AKAS](AKAS.md) · [Baseline phase rates and latency](BASELINES.md)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

Every shape is shown for prefill, decode and complete-burst output throughput. Phase columns display each runtime's reported timing boundary, labeled by its protocol. They do not calculate cross-protocol speedups. Each column links to its full profile and retained settings; failed profiles carry † even when a particular shape completed. Missing values are not replaced by zero.

## lfm2.5-230m / bf16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-lfm2.5-230m-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-lfm2.5-230m-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 8,918.2 | 9,037.3 | 3,032.4 | 3,003.6 | 2,889.6 | 2,050.4 |
| 1 | 2048 | 8,612.2 | 9,135.0 | 2,124.8 | 1,895.4 | 115,655.1 | 104,327.8 |
| 4 | 32 | 453.9 | 450.0 | 423.9 | 424.2 | 6,944.2 | 8,551.8 |
| 4 | 2048 | 7,876.3 | 7,751.6 | 771.0 | 741.5 | 117,833.0 | 105,143.8 |
| 16 | 32 | 35,601.2 | 35,578.9 | 3,320.9 | 3,257.5 | 22,006.7 | 27,579.2 |
| 16 | 2048 | 29,971.5 | 29,987.2 | 2,524.5 | 2,255.5 | 106,718.4 | 99,196.6 |
| 64 | 32 | 8,927.5 | 10,058.0 | 3,693.2 | 3,681.3 | 52,475.6 | 60,482.7 |
| 64 | 2048 | 56,574.6 | 57,379.1 | — | — | 87,542.5 | 97,806.0 |
| 256 | 32 | 23,772.3 | 23,624.3 | 4,287.5 | 4,248.3 | 91,362.6 | 67,298.5 |
| 256 | 2048 | 55,331.5 | 56,021.7 | — | — | 40,474.9 | 96,963.0 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-lfm2.5-230m-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-lfm2.5-230m-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 440.8 | 439.1 | 283.3 | 281.6 | 454.2 | 449.0 |
| 1 | 2048 | 404.3 | 404.0 | 248.6 | 230.6 | 441.4 | 436.6 |
| 4 | 32 | 1,104.9 | 1,107.5 | 814.1 | 811.6 | 1,745.7 | 1,697.9 |
| 4 | 2048 | 1,306.8 | 1,311.4 | 701.4 | 647.3 | 1,316.1 | 1,247.8 |
| 16 | 32 | 2,899.3 | 2,892.0 | 1,544.6 | 1,521.7 | 6,792.7 | 6,462.8 |
| 16 | 2048 | 2,246.7 | 2,283.3 | 1,029.0 | 918.3 | 2,748.2 | 2,431.5 |
| 64 | 32 | 12,856.6 | 12,792.6 | 2,070.5 | 2,030.6 | 20,705.2 | 20,230.8 |
| 64 | 2048 | 3,898.0 | 4,037.8 | — | — | 3,913.7 | 3,177.8 |
| 256 | 32 | 22,946.6 | 23,066.2 | 3,401.2 | 3,328.3 | 37,002.7 | 41,130.2 |
| 256 | 2048 | 4,336.5 | 4,626.7 | — | — | 5,756.3 | 3,410.1 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | [vllm cuda default / native call](BASELINES.md#baseline-lfm2.5-230m-vllm-cuda-bf16-ar) | [sglang cuda default / native call](BASELINES.md#baseline-lfm2.5-230m-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 171.6 | 164.9 | 128.5 | 151.9 | 439.9 | 426.6 |
| 1 | 2048 | 231.9 | 237.7 | 86.8 | 78.5 | 418.0 | 411.9 |
| 4 | 32 | 690.2 | 689.0 | 553.0 | 551.9 | 1,650.8 | 1,624.5 |
| 4 | 2048 | 358.3 | 354.5 | 45.1 | 43.3 | 1,224.1 | 1,165.7 |
| 16 | 32 | 2,863.4 | 2,856.2 | 1,393.4 | 1,372.2 | 6,330.7 | 6,126.0 |
| 16 | 2048 | 1,025.2 | 1,033.0 | 136.9 | 122.3 | 1,421.1 | 2,344.2 |
| 64 | 32 | 9,507.8 | 9,764.2 | 1,828.5 | 1,796.8 | 18,992.9 | 18,681.1 |
| 64 | 2048 | 1,861.0 | 1,906.3 | — | — | 1,856.5 | 3,125.9 |
| 256 | 32 | 18,602.8 | 18,657.7 | 2,856.9 | 2,801.5 | 34,137.9 | 37,332.0 |
| 256 | 2048 | 1,930.6 | 1,999.8 | — | — | 1,928.1 | 3,397.3 |

## lfm2.5-230m / f16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 8,362.9 | 8,417.8 | 1,932.7 | 1,922.7 | 5,315.6 | 3,512.2 |
| 1 | 2048 | 8,104.4 | 8,583.2 | 1,725.5 | 1,566.8 | 63,531.5 | 45,135.0 |
| 4 | 32 | 528.0 | 383.6 | 448.3 | 431.8 | 16,604.0 | 7,100.1 |
| 4 | 2048 | 7,641.6 | 7,525.7 | 758.4 | 722.3 | 63,447.3 | 42,024.3 |
| 16 | 32 | 38,653.2 | 38,229.5 | 3,352.9 | 3,352.2 | 30,378.5 | 17,479.2 |
| 16 | 2048 | 29,097.1 | 29,097.0 | 2,477.8 | 2,227.9 | 174,600.6 | 126,761.1 |
| 64 | 32 | 8,222.2 | 12,369.5 | 4,636.4 | 4,531.3 | 30,160.2 | 15,906.2 |
| 64 | 2048 | 54,864.5 | 55,328.3 | — | — | 465,662.9 | 372,549.9 |
| 256 | 32 | 26,430.9 | 25,142.2 | 4,327.4 | 4,091.6 | 25,135.7 | 17,363.6 |
| 256 | 2048 | 53,532.9 | 53,869.2 | — | — | 1,215,219.0 | 1,045,548.1 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 443.9 | 444.4 | 261.1 | 258.4 | 407.5 | 375.7 |
| 1 | 2048 | 407.2 | 406.6 | 242.1 | 212.0 | 387.9 | 359.5 |
| 4 | 32 | 818.0 | 818.7 | 229.7 | 225.0 | 1,240.4 | 1,158.5 |
| 4 | 2048 | 883.5 | 888.5 | 225.3 | 214.7 | 979.9 | 917.0 |
| 16 | 32 | 3,089.1 | 3,103.6 | 874.3 | 863.5 | 2,506.0 | 2,205.0 |
| 16 | 2048 | 2,340.8 | 2,390.4 | 720.5 | 666.9 | 1,316.4 | 1,083.7 |
| 64 | 32 | 12,863.4 | 12,909.5 | 2,450.0 | 2,414.7 | 3,385.7 | 3,362.1 |
| 64 | 2048 | 3,774.3 | 3,970.9 | — | — | 1,630.5 | 1,365.9 |
| 256 | 32 | 22,873.7 | 23,139.3 | 3,344.5 | 3,326.5 | 3,677.5 | 3,718.5 |
| 256 | 2048 | 4,026.3 | 4,438.1 | — | — | 2,988.6 | 2,548.1 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-f16-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 182.3 | 174.8 | 115.0 | 153.5 | 397.6 | 365.2 |
| 1 | 2048 | 226.7 | 232.3 | 74.8 | 67.2 | 352.9 | 318.5 |
| 4 | 32 | 593.0 | 536.6 | 205.0 | 200.5 | 1,193.3 | 1,089.0 |
| 4 | 2048 | 310.9 | 308.4 | 39.2 | 37.4 | 771.1 | 673.5 |
| 16 | 32 | 3,052.0 | 3,065.3 | 826.8 | 817.2 | 2,091.5 | 1,829.5 |
| 16 | 2048 | 1,027.0 | 1,036.3 | 127.6 | 115.3 | 999.0 | 841.5 |
| 64 | 32 | 9,299.0 | 10,302.1 | 2,179.1 | 2,145.6 | 1,642.5 | 1,608.3 |
| 64 | 2048 | 1,803.4 | 1,855.1 | — | — | 661.6 | 611.1 |
| 256 | 32 | 18,926.7 | 18,931.4 | 2,821.4 | 2,782.6 | 1,393.9 | 1,381.8 |
| 256 | 2048 | 1,833.8 | 1,921.0 | — | — | 467.4 | 442.5 |

## lfm2.5-230m / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 9,633.1 | 10,135.7 | 1,599.7 | 1,600.0 | 7,574.0 | 4,018.6 |
| 1 | 2048 | 9,573.1 | 10,246.2 | 1,446.0 | 1,348.3 | 63,568.9 | 47,682.2 |
| 4 | 32 | 504.2 | 502.8 | 364.4 | 353.5 | 18,140.6 | 9,405.5 |
| 4 | 2048 | 6,845.6 | 6,733.5 | 580.9 | 561.4 | 63,711.8 | 48,778.2 |
| 16 | 32 | 34,510.1 | 34,325.7 | 2,521.2 | 2,514.1 | 32,940.9 | 21,445.0 |
| 16 | 2048 | 26,057.8 | 26,050.2 | 1,986.4 | 1,826.5 | 192,085.2 | 142,506.7 |
| 64 | 32 | 12,264.9 | 12,061.6 | 3,664.8 | 3,652.0 | 27,146.3 | 20,201.6 |
| 64 | 2048 | 49,075.6 | 49,970.5 | — | — | 524,760.3 | 407,718.1 |
| 256 | 32 | 19,193.5 | 24,337.7 | 3,312.3 | 3,318.8 | 29,625.8 | 22,380.7 |
| 256 | 2048 | 47,825.1 | 48,749.5 | — | — | 1,414,231.1 | 1,118,218.7 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 974.0 | 983.0 | 432.0 | 442.4 | 752.7 | 678.3 |
| 1 | 2048 | 809.2 | 809.3 | 342.2 | 313.7 | 687.4 | 637.9 |
| 4 | 32 | 1,094.6 | 1,081.5 | 177.9 | 182.3 | 2,146.8 | 1,706.2 |
| 4 | 2048 | 1,157.2 | 1,165.6 | 180.2 | 175.7 | 1,528.1 | 1,259.5 |
| 16 | 32 | 4,105.1 | 4,087.1 | 707.0 | 705.1 | 3,856.0 | 2,515.5 |
| 16 | 2048 | 2,736.5 | 2,811.0 | 610.2 | 571.7 | 1,660.4 | 1,207.7 |
| 64 | 32 | 13,412.7 | 13,480.8 | 1,906.6 | 1,892.4 | 4,770.9 | 3,658.8 |
| 64 | 2048 | 3,762.9 | 3,956.1 | — | — | 1,903.2 | 1,498.9 |
| 256 | 32 | 19,713.3 | 19,980.6 | 2,481.4 | 2,470.9 | 5,320.0 | 3,873.4 |
| 256 | 2048 | 3,871.0 | 4,361.6 | — | — | 3,493.7 | 2,739.7 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_0-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 236.2 | 243.5 | 154.6 | 196.0 | 728.8 | 641.6 |
| 1 | 2048 | 345.1 | 358.7 | 71.6 | 66.5 | 580.8 | 522.3 |
| 4 | 32 | 713.1 | 706.9 | 159.7 | 162.6 | 2,007.2 | 1,576.6 |
| 4 | 2048 | 313.0 | 309.8 | 30.3 | 29.3 | 1,073.2 | 876.7 |
| 16 | 32 | 4,017.0 | 3,999.3 | 665.5 | 663.7 | 2,961.8 | 2,076.2 |
| 16 | 2048 | 1,024.0 | 1,034.0 | 103.3 | 95.3 | 1,184.6 | 922.5 |
| 64 | 32 | 10,598.0 | 10,601.4 | 1,698.9 | 1,687.1 | 1,880.8 | 1,686.8 |
| 64 | 2048 | 1,695.8 | 1,751.4 | — | — | 720.9 | 649.0 |
| 256 | 32 | 15,783.8 | 16,686.3 | 2,103.8 | 2,097.0 | 1,580.9 | 1,419.5 |
| 256 | 2048 | 1,692.4 | 1,799.6 | — | — | 496.0 | 460.0 |

## lfm2.5-230m / q4_k_m / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 8,547.2 | 8,561.2 | 1,449.3 | 1,452.0 | 6,632.1 | 3,465.8 |
| 1 | 2048 | 8,046.2 | 8,529.2 | 1,319.0 | 1,238.5 | 60,120.4 | 43,773.8 |
| 4 | 32 | 479.6 | 475.6 | 272.1 | 322.4 | 16,684.0 | 8,171.1 |
| 4 | 2048 | 5,394.8 | 5,418.2 | 527.0 | 513.9 | 59,963.7 | 42,753.5 |
| 16 | 32 | 27,570.3 | 27,277.0 | 2,294.7 | 2,276.4 | 31,236.7 | 18,705.9 |
| 16 | 2048 | 20,828.8 | 21,153.8 | 1,831.6 | 1,703.1 | 183,798.7 | 126,078.3 |
| 64 | 32 | 10,512.9 | 10,812.4 | 3,318.7 | 2,903.8 | 24,070.6 | 17,399.1 |
| 64 | 2048 | 39,640.4 | 40,457.8 | — | — | 505,065.2 | 373,141.7 |
| 256 | 32 | 20,518.2 | 16,734.2 | 3,022.7 | 3,045.3 | 27,065.3 | 19,717.9 |
| 256 | 2048 | 38,824.6 | 39,506.5 | — | — | 1,372,941.7 | 1,067,970.3 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 894.9 | 902.3 | 301.7 | 281.3 | 727.1 | 648.6 |
| 1 | 2048 | 744.8 | 745.1 | 241.7 | 236.1 | 663.5 | 615.4 |
| 4 | 32 | 932.5 | 895.4 | 167.3 | 168.6 | 2,064.3 | 1,539.1 |
| 4 | 2048 | 1,028.8 | 1,048.7 | 167.4 | 164.4 | 1,477.8 | 1,146.8 |
| 16 | 32 | 3,577.7 | 3,573.9 | 645.5 | 642.5 | 3,737.8 | 2,167.3 |
| 16 | 2048 | 2,547.7 | 2,639.6 | 565.4 | 538.5 | 1,604.4 | 1,083.3 |
| 64 | 32 | 11,642.6 | 11,804.6 | 1,773.2 | 1,777.7 | 4,597.6 | 3,554.4 |
| 64 | 2048 | 3,635.8 | 3,827.7 | — | — | 1,838.3 | 1,389.6 |
| 256 | 32 | 17,639.6 | 18,037.3 | 2,330.6 | 2,346.8 | 5,157.2 | 3,836.3 |
| 256 | 2048 | 3,734.6 | 4,212.7 | — | — | 3,374.9 | 2,602.1 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-cuda-q4_k_m-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-lfm2.5-230m-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 222.5 | 229.5 | 116.8 | 155.7 | 704.9 | 617.9 |
| 1 | 2048 | 301.1 | 311.8 | 61.6 | 58.4 | 559.8 | 499.1 |
| 4 | 32 | 630.8 | 612.1 | 146.0 | 150.1 | 1,923.0 | 1,414.7 |
| 4 | 2048 | 254.4 | 256.5 | 27.6 | 26.9 | 1,034.3 | 789.7 |
| 16 | 32 | 3,491.7 | 3,486.9 | 607.6 | 604.6 | 2,881.1 | 1,845.5 |
| 16 | 2048 | 863.8 | 883.2 | 95.3 | 89.0 | 1,171.3 | 838.1 |
| 64 | 32 | 9,174.2 | 9,330.8 | 1,575.1 | 1,552.2 | 1,863.7 | 1,660.4 |
| 64 | 2048 | 1,478.2 | 1,527.4 | — | — | 713.4 | 625.0 |
| 256 | 32 | 14,613.0 | 14,296.6 | 1,966.8 | 1,980.7 | 1,555.4 | 1,411.1 |
| 256 | 2048 | 1,475.4 | 1,561.2 | — | — | 486.9 | 451.0 |

## qwen3.5-0.8b / bf16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 3,848.2 | 3,786.3 | 252.6 | 252.8 | 924.1 | 1,189.1 |
| 1 | 2048 | 3,773.5 | 2,183.7 | 237.1 | 233.0 | 31,334.5 | 29,066.9 |
| 4 | 32 | 670.7 | 677.0 | 49.2 | 49.5 | 1,728.5 | 4,312.4 |
| 4 | 2048 | 2,668.7 | 2,326.8 | — | — | 27,927.1 | 28,037.5 |
| 16 | 32 | 10,669.6 | 10,911.9 | 201.2 | 197.5 | 6,154.3 | 15,963.8 |
| 16 | 2048 | 8,974.5 | 7,255.9 | — | — | 27,720.3 | 28,738.6 |
| 64 | 32 | 7,243.8 | 7,135.4 | — | — | 12,999.2 | 21,238.8 |
| 64 | 2048 | 13,531.0 | 9,543.5 | — | — | 23,233.8 | 28,451.9 |
| 256 | 32 | 8,044.8 | 8,753.4 | — | — | 13,822.8 | 21,967.4 |
| 256 | 2048 | — | — | — | — | 15,968.0 | 27,592.6 |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-ar) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 143.9 | 131.5 | 71.4 | 70.0 | 146.9 | 148.8 |
| 1 | 2048 | 132.9 | 47.6 | 41.0 | 31.6 | 141.3 | 144.2 |
| 4 | 32 | 459.8 | 426.8 | 37.3 | 37.2 | 523.9 | 540.5 |
| 4 | 2048 | 489.0 | 181.9 | — | — | 421.5 | 423.5 |
| 16 | 32 | 1,461.4 | 1,304.6 | 144.4 | 142.6 | 1,656.7 | 1,682.3 |
| 16 | 2048 | 1,409.0 | 661.8 | — | — | 862.6 | 826.2 |
| 64 | 32 | 3,152.8 | 2,897.2 | — | — | 3,360.7 | 3,369.1 |
| 64 | 2048 | 2,404.2 | 1,046.4 | — | — | 1,136.4 | 1,050.9 |
| 256 | 32 | 4,063.3 | 3,755.9 | — | — | 4,466.2 | 4,318.5 |
| 256 | 2048 | — | — | — | — | 1,857.1 | 1,107.3 |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | [vllm cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-ar) | [sglang cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 101.4 | 88.2 | 49.4 | 50.6 | 142.4 | 145.2 |
| 1 | 2048 | 85.4 | 35.5 | 10.9 | 10.0 | 132.8 | 134.4 |
| 4 | 32 | 395.2 | 371.2 | 31.6 | 31.5 | 490.6 | 527.3 |
| 4 | 2048 | 124.6 | 81.1 | — | — | 388.4 | 395.5 |
| 16 | 32 | 1,423.8 | 1,276.4 | 123.2 | 121.6 | 1,563.2 | 1,648.2 |
| 16 | 2048 | 402.1 | 270.0 | — | — | 790.4 | 801.5 |
| 64 | 32 | 2,863.6 | 2,649.0 | — | — | 3,236.2 | 3,263.9 |
| 64 | 2048 | 626.9 | 381.0 | — | — | 987.1 | 1,044.2 |
| 256 | 32 | 3,632.9 | 3,416.2 | — | — | 4,333.2 | 4,260.9 |
| 256 | 2048 | — | — | — | — | 907.5 | 1,108.9 |

## qwen3.5-0.8b / bf16 / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-mtp) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 250.1 | 265.8 | 154.6 | 165.1 | 823.9 † | 933.8 † |
| 1 | 2048 | 2,882.0 | 1,747.0 | 219.4 | 215.5 | 31,068.1 † | 25,544.1 † |
| 4 | 32 | 414.9 | 419.0 | 45.8 | 44.9 | 2,706.7 † | 3,567.8 † |
| 4 | 2048 | 2,435.9 | 2,115.1 | — | — | 26,241.9 † | 24,898.7 † |
| 16 | 32 | 9,746.1 | 10,080.4 | 187.6 | 188.3 | 5,684.6 † | 13,282.7 † |
| 16 | 2048 | 8,250.8 | 6,600.1 | — | — | 25,624.5 † | 26,068.6 † |
| 64 | 32 | 4,456.0 | 4,471.4 | — | — | 14,600.0 † | 19,014.3 † |
| 64 | 2048 | 12,589.0 | 8,569.9 | — | — | 20,328.5 † | 26,061.6 † |
| 256 | 32 | 6,199.5 | 6,139.3 | — | — | 11,781.8 † | 19,726.8 † |
| 256 | 2048 | — | — | — | — | 16,618.7 † | 16,862.5 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | [vllm cuda default / request-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-mtp) | [sglang cuda default / sglang-scheduler-phase-v1](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 137.1 | 130.5 | 15.4 | 16.2 | 200.5 † | 181.8 † |
| 1 | 2048 | 196.5 | 77.0 | 15.9 | 14.8 | 190.7 † | 187.1 † |
| 4 | 32 | 470.9 | 447.0 | 53.8 | 55.3 | 751.8 † | 625.0 † |
| 4 | 2048 | 690.6 | 283.6 | — | — | 576.9 † | 523.4 † |
| 16 | 32 | 1,481.9 | 1,379.8 | 139.4 | 133.2 | 1,867.5 † | 1,500.4 † |
| 16 | 2048 | 1,741.3 | 867.5 | — | — | 986.1 † | 844.9 † |
| 64 | 32 | 3,959.8 | 3,723.0 | — | — | 4,131.3 † | 2,935.2 † |
| 64 | 2048 | 3,046.6 | 1,520.8 | — | — | 1,229.7 † | 999.4 † |
| 256 | 32 | 4,656.7 | 4,420.6 | — | — | 5,279.1 † | 3,396.3 † |
| 256 | 2048 | — | — | — | — | 3,244.6 † | 1,500.2 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | [vllm cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-vllm-cuda-bf16-mtp) | [sglang cuda default / native call](BASELINES.md#baseline-qwen3.5-0.8b-sglang-cuda-bf16-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 99.8 | 96.1 | 15.0 | 15.0 | 190.3 † | 174.5 † |
| 1 | 2048 | 94.3 | 45.4 | 7.4 | 7.1 | 174.8 † | 168.4 † |
| 4 | 32 | 369.1 | 355.1 | 41.8 | 42.5 | 707.8 † | 602.6 † |
| 4 | 2048 | 124.9 | 90.4 | — | — | 487.1 † | 449.2 † |
| 16 | 32 | 1,438.5 | 1,344.3 | 118.3 | 113.9 | 1,737.8 † | 1,466.1 † |
| 16 | 2048 | 398.6 | 280.3 | — | — | 799.3 † | 746.2 † |
| 64 | 32 | 3,260.9 | 3,101.6 | — | — | 3,928.6 † | 2,843.0 † |
| 64 | 2048 | 626.3 | 396.9 | — | — | 934.0 † | 962.9 † |
| 256 | 32 | 3,946.4 | 3,771.2 | — | — | 5,120.4 † | 3,315.3 † |
| 256 | 2048 | — | — | — | — | 968.5 † | 941.8 † |

## qwen3.5-0.8b / f16 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 2,488.1 | 2,461.9 | 251.6 | 252.6 | 2,636.8 † | 2,117.4 † |
| 1 | 2048 | 2,478.7 | 1,666.5 | 237.1 | 233.3 | 19,112.0 † | 11,683.0 † |
| 4 | 32 | 701.2 | 701.7 | 49.9 | 47.1 | 5,674.3 † | 3,942.2 † |
| 4 | 2048 | 1,964.3 | 1,782.3 | — | — | 32,924.5 † | 21,510.9 † |
| 16 | 32 | 7,771.4 | 8,185.9 | 199.9 | 197.9 | 10,120.0 † | 6,213.1 † |
| 16 | 2048 | 6,899.3 | 5,868.3 | — | — | 119,751.6 † | 76,799.4 † |
| 64 | 32 | 6,553.5 | 6,474.5 | — | — | 13,085.8 † | 4,241.4 † |
| 64 | 2048 | 11,108.8 | 8,263.2 | — | — | 362,758.8 † | 245,588.9 † |
| 256 | 32 | 7,403.3 | 7,333.2 | — | — | 9,951.0 † | 7,280.7 † |
| 256 | 2048 | — | — | — | — | 713,479.1 † | 496,155.0 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-f16-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 145.2 | 132.7 | 69.8 | 72.1 | 132.2 † | 119.6 † |
| 1 | 2048 | 133.1 | 47.7 | 41.1 | 32.0 | 131.4 † | 116.7 † |
| 4 | 32 | 302.3 | 289.5 | 37.5 | 37.3 | 388.5 † | 330.1 † |
| 4 | 2048 | 311.5 | 150.1 | — | — | 295.2 † | 240.2 † |
| 16 | 32 | 972.9 | 990.1 | 144.4 | 143.9 | 825.8 † | 567.7 † |
| 16 | 2048 | 990.5 | 554.1 | — | — | 467.7 † | 315.0 † |
| 64 | 32 | 2,535.1 | 2,388.1 | — | — | 996.7 † | 737.5 † |
| 64 | 2048 | 2,028.3 | 970.2 | — | — | 550.2 † | 386.4 † |
| 256 | 32 | 3,615.7 | 3,383.4 | — | — | 1,064.2 † | 777.6 † |
| 256 | 2048 | — | — | — | — | 1,059.4 † | 738.9 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-f16-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-f16-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 94.5 | 97.5 | 51.2 | 48.7 | 131.3 † | 118.5 † |
| 1 | 2048 | 71.9 | 32.9 | 10.9 | 10.0 | 118.8 † | 101.1 † |
| 4 | 32 | 274.8 | 264.3 | 31.8 | 31.4 | 384.3 † | 325.5 † |
| 4 | 2048 | 88.3 | 64.1 | — | — | 273.7 † | 218.6 † |
| 16 | 32 | 950.6 | 968.4 | 123.1 | 122.6 | 817.8 † | 563.5 † |
| 16 | 2048 | 301.1 | 221.4 | — | — | 434.0 † | 293.2 † |
| 64 | 32 | 2,328.1 | 2,202.2 | — | — | 986.0 † | 732.0 † |
| 64 | 2048 | 518.3 | 338.0 | — | — | 467.1 † | 331.6 † |
| 256 | 32 | 3,244.9 | 3,054.9 | — | — | 1,044.2 † | 764.7 † |
| 256 | 2048 | — | — | — | — | 486.9 † | 341.8 † |

## qwen3.5-0.8b / f16 / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 315.2 | 310.2 | 161.8 | 109.6 |
| 1 | 2048 | 2,058.2 | 1,396.0 | 220.1 | 216.6 |
| 4 | 32 | 308.8 | 414.7 | 46.5 | 46.6 |
| 4 | 2048 | 1,828.7 | 1,631.7 | — | — |
| 16 | 32 | 7,151.2 | 7,421.0 | 189.1 | 191.6 |
| 16 | 2048 | 6,450.2 | 5,360.9 | — | — |
| 64 | 32 | 4,288.7 | 3,476.9 | — | — |
| 64 | 2048 | 10,349.6 | 7,490.9 | — | — |
| 256 | 32 | 5,303.0 | 5,237.0 | — | — |
| 256 | 2048 | — | — | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 105.3 | 102.6 | 16.3 | 16.3 |
| 1 | 2048 | 134.1 | 65.0 | 15.8 | 14.9 |
| 4 | 32 | 383.4 | 371.0 | 55.8 | 55.8 |
| 4 | 2048 | 466.8 | 235.5 | — | — |
| 16 | 32 | 1,139.4 | 1,083.7 | 139.9 | 141.9 |
| 16 | 2048 | 1,226.6 | 709.0 | — | — |
| 64 | 32 | 3,224.3 | 3,076.9 | — | — |
| 64 | 2048 | 2,597.2 | 1,392.4 | — | — |
| 256 | 32 | 3,990.5 | 3,821.9 | — | — |
| 256 | 2048 | — | — | — | — |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 82.2 | 73.7 | 14.7 | 15.1 |
| 1 | 2048 | 65.9 | 37.4 | 7.4 | 7.1 |
| 4 | 32 | 294.3 | 305.1 | 43.2 | 43.2 |
| 4 | 2048 | 92.0 | 71.3 | — | — |
| 16 | 32 | 1,104.1 | 1,053.5 | 118.8 | 120.5 |
| 16 | 2048 | 304.0 | 228.1 | — | — |
| 64 | 32 | 2,732.1 | 2,535.7 | — | — |
| 64 | 2048 | 518.7 | 351.1 | — | — |
| 256 | 32 | 3,380.9 | 3,253.7 | — | — |
| 256 | 2048 | — | — | — | — |

## qwen3.5-0.8b / q4_0 / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 3,201.4 | 3,162.6 | 257.0 | 253.5 | 3,739.6 † | 2,577.9 † |
| 1 | 2048 | 3,152.8 | 1,947.3 | 240.7 | 236.8 | 18,641.9 † | 12,151.3 † |
| 4 | 32 | 545.2 | 547.3 | 47.4 | 48.3 | 7,192.6 † | 4,493.9 † |
| 4 | 2048 | 1,447.6 | 1,326.9 | — | — | 33,187.6 † | 22,545.4 † |
| 16 | 32 | 5,843.5 | 5,861.3 | 194.4 | 195.8 | 10,485.1 † | 6,636.4 † |
| 16 | 2048 | 5,266.2 | 4,617.3 | — | — | 119,770.5 † | 80,657.9 † |
| 64 | 32 | 5,516.7 | 4,342.8 | — | — | 11,174.0 † | 8,828.4 † |
| 64 | 2048 | 8,872.7 | 6,963.0 | — | — | 369,641.7 † | 252,305.6 † |
| 256 | 32 | 5,539.3 | 5,874.1 | — | — | 10,332.6 † | 7,447.6 † |
| 256 | 2048 | — | — | — | — | 743,784.9 † | 509,747.0 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_0-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 227.4 | 200.6 | 81.6 | 78.5 | 263.5 † | 219.7 † |
| 1 | 2048 | 201.5 | 55.9 | 45.9 | 33.8 | 252.4 † | 213.3 † |
| 4 | 32 | 389.0 | 367.3 | 37.7 | 38.8 | 671.0 † | 479.9 † |
| 4 | 2048 | 410.8 | 170.1 | — | — | 416.3 † | 314.4 † |
| 16 | 32 | 1,283.9 | 1,225.4 | 143.8 | 149.9 | 1,015.8 † | 635.7 † |
| 16 | 2048 | 1,237.2 | 627.5 | — | — | 512.8 † | 340.5 † |
| 64 | 32 | 2,607.4 | 2,446.8 | — | — | 1,068.1 † | 772.6 † |
| 64 | 2048 | 2,072.2 | 989.8 | — | — | 571.1 † | 399.1 † |
| 256 | 32 | 3,043.4 | 2,868.5 | — | — | 1,108.1 † | 784.9 † |
| 256 | 2048 | — | — | — | — | 1,104.0 † | 757.8 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_0-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_0-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 136.9 | 126.8 | 56.9 | 57.2 | 260.1 † | 216.2 † |
| 1 | 2048 | 100.0 | 38.5 | 11.4 | 10.3 | 207.8 † | 167.3 † |
| 4 | 32 | 332.3 | 316.7 | 31.7 | 32.5 | 659.7 † | 468.0 † |
| 4 | 2048 | 74.2 | 55.9 | — | — | 374.3 † | 282.1 † |
| 16 | 32 | 1,226.1 | 1,173.2 | 122.1 | 126.6 | 1,004.9 † | 631.0 † |
| 16 | 2048 | 260.4 | 198.2 | — | — | 477.5 † | 317.3 † |
| 64 | 32 | 2,348.2 | 2,159.5 | — | — | 1,057.1 † | 764.8 † |
| 64 | 2048 | 438.2 | 303.0 | — | — | 490.4 † | 343.8 † |
| 256 | 32 | 2,694.4 | 2,574.3 | — | — | 1,088.3 † | 772.2 † |
| 256 | 2048 | — | — | — | — | 512.7 † | 352.1 † |

## qwen3.5-0.8b / q4_0 / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 271.8 | 271.2 | 127.6 | 163.8 |
| 1 | 2048 | 2,726.6 | 1,673.9 | 223.8 | 219.5 |
| 4 | 32 | 369.3 | 362.1 | 43.2 | 44.5 |
| 4 | 2048 | 1,363.4 | 1,221.8 | — | — |
| 16 | 32 | 5,430.1 | 5,445.7 | 183.0 | 182.6 |
| 16 | 2048 | 4,970.2 | 4,201.3 | — | — |
| 64 | 32 | 3,718.2 | 3,573.1 | — | — |
| 64 | 2048 | 8,335.5 | 6,351.3 | — | — |
| 256 | 32 | 4,236.3 | 3,842.1 | — | — |
| 256 | 2048 | — | — | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 133.7 | 114.5 | 16.6 | 17.3 |
| 1 | 2048 | 144.4 | 58.4 | 13.1 | 12.6 |
| 4 | 32 | 470.3 | 444.2 | 54.3 | 57.5 |
| 4 | 2048 | 601.5 | 269.1 | — | — |
| 16 | 32 | 1,243.0 | 1,290.0 | 137.1 | 140.7 |
| 16 | 2048 | 1,401.0 | 749.2 | — | — |
| 64 | 32 | 2,737.6 | 2,524.9 | — | — |
| 64 | 2048 | 2,242.9 | 1,261.7 | — | — |
| 256 | 32 | 3,077.3 | 2,927.3 | — | — |
| 256 | 2048 | — | — | — | — |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 87.9 | 86.9 | 16.3 | 15.8 |
| 1 | 2048 | 78.5 | 37.7 | 6.8 | 6.6 |
| 4 | 32 | 358.8 | 342.0 | 41.5 | 43.7 |
| 4 | 2048 | 74.7 | 59.6 | — | — |
| 16 | 32 | 1,184.5 | 1,226.9 | 116.3 | 118.8 |
| 16 | 2048 | 254.6 | 194.8 | — | — |
| 64 | 32 | 2,327.4 | 2,160.2 | — | — |
| 64 | 2048 | 423.4 | 302.5 | — | — |
| 256 | 32 | 2,621.7 | 2,475.2 | — | — |
| 256 | 2048 | — | — | — | — |

## qwen3.5-0.8b / q4_k_m / ar

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 2,192.0 | 2,167.8 | 253.1 | 252.6 | 3,597.1 † | 2,259.2 † |
| 1 | 2048 | 2,145.5 | 1,504.1 | 237.4 | 234.4 | 17,700.3 † | 11,317.8 † |
| 4 | 32 | 394.0 | 392.4 | 46.0 | 47.3 | 6,882.8 † | 4,299.8 † |
| 4 | 2048 | 805.7 | 765.3 | — | — | 32,045.2 † | 21,061.7 † |
| 16 | 32 | 3,215.8 | 3,210.1 | 194.7 | 190.0 | 7,549.1 † | 5,820.4 † |
| 16 | 2048 | 3,054.8 | 2,808.8 | — | — | 115,418.9 † | 75,003.0 † |
| 64 | 32 | 3,707.1 | 3,770.5 | — | — | 11,353.2 † | 8,675.7 † |
| 64 | 2048 | 5,476.6 | — | — | — | 359,008.9 † | 216,310.1 † |
| 256 | 32 | 3,859.3 | 3,870.9 | — | — | 9,865.4 † | 7,641.5 † |
| 256 | 2048 | — | — | — | — | 725,525.6 † | 489,870.2 † |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | [llama cuda default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_k_m-ar) | [llama vulkan default / llama-server-eval-v1](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 160.8 | 146.7 | 73.0 | 71.2 | 251.1 † | 212.3 † |
| 1 | 2048 | 146.4 | 51.8 | 42.6 | 32.4 | 241.2 † | 207.7 † |
| 4 | 32 | 262.8 | 239.3 | 37.1 | 38.2 | 630.4 † | 448.7 † |
| 4 | 2048 | 275.3 | 141.9 | — | — | 395.8 † | 292.4 † |
| 16 | 32 | 873.6 | 891.4 | 144.0 | 146.6 | 984.2 † | 594.7 † |
| 16 | 2048 | 900.6 | 528.6 | — | — | 496.9 † | 318.8 † |
| 64 | 32 | 1,985.1 | 1,908.0 | — | — | 1,074.8 † | 763.1 † |
| 64 | 2048 | 1,635.9 | — | — | — | 574.0 † | 360.6 † |
| 256 | 32 | 2,375.3 | 2,282.4 | — | — | 1,094.3 † | 783.2 † |
| 256 | 2048 | — | — | — | — | 1,071.8 † | 730.9 † |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | [llama cuda default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-cuda-q4_k_m-ar) | [llama vulkan default / HTTP](BASELINES.md#baseline-qwen3.5-0.8b-llama-vulkan-q4_k_m-ar) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 92.8 | 90.4 | 51.8 | 51.4 | 247.8 † | 208.0 † |
| 1 | 2048 | 70.2 | 33.6 | 11.0 | 10.1 | 198.5 † | 160.6 † |
| 4 | 32 | 226.8 | 209.1 | 31.0 | 32.0 | 620.2 † | 441.7 † |
| 4 | 2048 | 42.6 | 35.8 | — | — | 357.3 † | 261.4 † |
| 16 | 32 | 824.1 | 839.6 | 122.4 | 123.7 | 974.5 † | 589.8 † |
| 16 | 2048 | 157.7 | 132.0 | — | — | 463.1 † | 297.6 † |
| 64 | 32 | 1,762.9 | 1,705.5 | — | — | 1,063.0 † | 755.8 † |
| 64 | 2048 | 283.4 | — | — | — | 491.7 † | 313.5 † |
| 256 | 32 | 2,072.6 | 2,002.8 | — | — | 1,074.5 † | 770.2 † |
| 256 | 2048 | — | — | — | — | 500.0 † | 338.4 † |

## qwen3.5-0.8b / q4_k_m / mtp

### Prefill tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 248.0 | 223.3 | 99.1 | 159.3 |
| 1 | 2048 | 1,891.2 | 1,324.7 | 221.9 | 217.7 |
| 4 | 32 | 271.5 | 228.6 | 44.6 | 44.0 |
| 4 | 2048 | 774.3 | 735.5 | — | — |
| 16 | 32 | 3,100.9 | 3,096.0 | 184.3 | 182.3 |
| 16 | 2048 | 2,937.6 | 2,681.6 | — | — |
| 64 | 32 | 2,346.1 | 2,760.4 | — | — |
| 64 | 2048 | 5,240.3 | — | — | — |
| 256 | 32 | 2,922.4 | 2,904.1 | — | — |
| 256 | 2048 | — | — | — | — |

### Decode tok/s — reported phase timing

| B | Prefix | [akas cuda default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | [akas cuda fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | [akas webgpu default / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | [akas webgpu fp8 / native-token-step-v2](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 89.4 | 87.1 | 15.3 | 15.3 |
| 1 | 2048 | 123.0 | 61.9 | 16.1 | 15.0 |
| 4 | 32 | 353.6 | 342.6 | 53.1 | 55.6 |
| 4 | 2048 | 446.5 | 235.7 | — | — |
| 16 | 32 | 1,066.1 | 1,023.7 | 140.8 | 137.6 |
| 16 | 2048 | 1,207.3 | 702.2 | — | — |
| 64 | 32 | 2,181.8 | 2,122.9 | — | — |
| 64 | 2048 | 1,829.5 | — | — | — |
| 256 | 32 | 2,473.5 | 2,405.6 | — | — |
| 256 | 2048 | — | — | — | — |

### Complete-burst output tok/s

| B | Prefix | [akas cuda default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | [akas cuda fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | [akas webgpu default / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | [akas webgpu fp8 / HTTP](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 65.1 | 69.0 | 15.1 | 14.3 |
| 1 | 2048 | 60.5 | 35.6 | 7.5 | 7.2 |
| 4 | 32 | 268.3 | 250.7 | 41.2 | 42.5 |
| 4 | 2048 | 43.7 | 38.5 | — | — |
| 16 | 32 | 988.9 | 952.4 | 119.0 | 116.5 |
| 16 | 2048 | 159.5 | 135.5 | — | — |
| 64 | 32 | 1,781.5 | 1,792.3 | — | — |
| 64 | 2048 | 278.1 | — | — | — |
| 256 | 32 | 2,054.7 | 2,005.9 | — | — |
| 256 | 2048 | — | — | — | — |

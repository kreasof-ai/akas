# Campaign benchmark report

UTC: 2026-10-15T22:28:44.060326Z → 2026-10-16T12:28:02.539059Z. Status: `configuration_budgets_exhausted`. Active time: 13.99 hours; per-configuration ceiling: 1.00 hours.

Source: `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8`; measured source tree: `0b85cd888e417e2bf9da99c10e56d41b4a5ac0c9258fa6458af16ac57b754c4e`.

Qualified AKAS deployments: **8/48**. Qualified baseline profiles: **10/18**.

- [Side-by-side tables for every requested shape](COMPARISON.md)
- [All AKAS configurations and per-shape measurements](AKAS.md)
- [All baseline profiles and per-shape measurements](BASELINES.md)
- [Every candidate attempt](CANDIDATES.md)
- [Full measurement CSV](measurements.csv) · [Candidate measurement CSV](candidate-measurements.csv) · [Candidate outcomes CSV](candidates.csv)
- [Selected kernel implementations and schedules](SCHEDULES.json) · [Portable report data](report.json) · [Campaign snapshot](campaign.json)

All throughput and displayed latency values are medians over the retained repetitions. AKAS native decode counts the outputs after the first token; burst throughput counts all outputs. AKAS burst timing includes loopback HTTP/SSE, vLLM and SGLang burst timing uses native engine calls, and llama.cpp burst timing uses concurrent HTTP requests. Phase rates have distinct protocols and cannot be compared as equivalent kernel timings. Weight labels describe source storage; execution precision is recorded per configuration and can differ between operations. Unqualified timing values remain visible with † and are excluded from qualified claims. — means no retained measurement, never zero throughput.

## Hardware and measurement provenance

| Hardware | Driver | Baseline/runtime | Version or revision | Source | Evidence |
| --- | --- | --- | --- | --- | --- |
| NVIDIA L4 |  | akas / lfm2.5-230m / bf16 / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / bf16 / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / f16 / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / f16 / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_0 / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_0 / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_k_m / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| NVIDIA L4 |  | akas / lfm2.5-230m / q4_k_m / ar | 0.1.0 | `3e90ad81dc812a17acc17e0acda34ca3b69ee3b8` | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |
| NVIDIA L4 | 595.91.07 | vllm / lfm2.5-230m / bf16 / ar | 0.31.0 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015181815641257-391d76bc](../../runs/20261015181815641257-391d76bc/run.json) |
| NVIDIA L4 | 595.91.07 | sglang / lfm2.5-230m / bf16 / ar | 0.5.21 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015190359153016-bcd95971](../../runs/20261015190359153016-bcd95971/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015192832378421-704bee0c](../../runs/20261015192832378421-704bee0c/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015193656299664-fe2eba27](../../runs/20261015193656299664-fe2eba27/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015194346054741-23180dda](../../runs/20261015194346054741-23180dda/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015195038132061-53344b1f](../../runs/20261015195038132061-53344b1f/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015201742219174-9ab41fef](../../runs/20261015201742219174-9ab41fef/run.json) |
| NVIDIA L4 | 595.91.07 | llama / lfm2.5-230m / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015202523676529-ef8b1d74](../../runs/20261015202523676529-ef8b1d74/run.json) |
| NVIDIA L4 | 595.91.07 | vllm / qwen3.5-0.8b / bf16 / ar | 0.31.0 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015191912117158-bf3b040b](../../runs/20261015191912117158-bf3b040b/run.json) |
| NVIDIA L4 | 595.91.07 | vllm / qwen3.5-0.8b / bf16 / mtp | 0.31.0 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015205432983160-cebe91d1](../../runs/20261015205432983160-cebe91d1/run.json) |
| NVIDIA L4 | 595.91.07 | sglang / qwen3.5-0.8b / bf16 / ar | 0.5.21 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015190536066752-9e49f8e2](../../runs/20261015190536066752-9e49f8e2/run.json) |
| NVIDIA L4 | 595.91.07 | sglang / qwen3.5-0.8b / bf16 / mtp | 0.5.21 | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015190958176695-723ea8a5](../../runs/20261015190958176695-723ea8a5/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015210200344123-b240f1f0](../../runs/20261015210200344123-b240f1f0/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015211120204768-ca9ae9c9](../../runs/20261015211120204768-ca9ae9c9/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015212004566075-2f3bc973](../../runs/20261015212004566075-2f3bc973/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / f16 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015212857450400-a5e9f4cb](../../runs/20261015212857450400-a5e9f4cb/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_0 / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015214155673053-00188f08](../../runs/20261015214155673053-00188f08/run.json) |
| NVIDIA L4 | 595.91.07 | llama / qwen3.5-0.8b / q4_k_m / ar | 620d2225cc436ab0b001e3200e590e3b286ef9ab | `b80c3599312b7ce568a5770381ea7a99a0412399` | [20261015215423483123-d676f19f](../../runs/20261015215423483123-d676f19f/run.json) |

Captured host: 4 logical CPUs; Linux-6.12.103-129.197.amzn2023.x86_64-x86_64-with-glibc2.39.

```text
index, uuid, name, driver_version, memory.total [MiB], power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
0, GPU-d777b838-7002-124f-aa67-71409daa8935, NVIDIA L4, 595.91.07, 23034 MiB, 72.00 W, 2040 MHz, 6251 MHz
```

Full dependency versions, runtime hyperparameters, checkpoint hashes, UTC timestamps, sample values and artifact checksums are retained in the linked run records and report.json. Selected schedules include each operation’s parameters, implementation, geometry and qualification evidence.

## Selected configurations

HTTP speedup below is against the first qualified AKAS candidate in the same configuration. External baselines are shown separately in the side-by-side tables. Peak VRAM is sampled whole-device usage.

| Configuration | KV dtype | Chunk | Prefill group | Replay | Trials allowance | HTTP speedup over own reference | Peak VRAM GiB | Evidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| lfm2.5-230m-bf16-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.059x | 7.6 | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |
| lfm2.5-230m-bf16-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.000x | 4.4 | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |
| lfm2.5-230m-f16-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.041x | 7.6 | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |
| lfm2.5-230m-f16-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.039x | 4.4 | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |
| lfm2.5-230m-q4_0-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.112x | 7.2 | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |
| lfm2.5-230m-q4_0-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.109x | 4.1 | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |
| lfm2.5-230m-q4_k_m-cuda-default-ar | float16 | 32 | 32 | True | 1 | 1.089x | 7.3 | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |
| lfm2.5-230m-q4_k_m-cuda-fp8-ar | fp8 | 32 | 32 | True | 1 | 1.078x | 4.1 | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |

## Complete requested AKAS coverage

| Model | Weights | Backend | KV request | Mode | Outcome | Time min | Selected run | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| lfm2.5-230m | bf16 | cuda | default | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-bf16-cuda-default-ar) | 60.0 | [20261016075804599992-469ebb3b](../../runs/20261016075804599992-469ebb3b/run.json) |  |
| lfm2.5-230m | bf16 | cuda | fp8 | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-bf16-cuda-fp8-ar) | 60.0 | [20261015231748568768-a50eb4e9](../../runs/20261015231748568768-a50eb4e9/run.json) |  |
| lfm2.5-230m | bf16 | webgpu | default | ar | [failed](AKAS.md#lfm2.5-230m-bf16-webgpu-default-ar) | 4.6 | — | worker exited with code -9 |
| lfm2.5-230m | bf16 | webgpu | fp8 | ar | [budget_exhausted](AKAS.md#lfm2.5-230m-bf16-webgpu-fp8-ar) | 60.1 | — | worker wall-clock deadline exceeded |
| lfm2.5-230m | f16 | cuda | default | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-f16-cuda-default-ar) | 60.0 | [20261016002444872634-811c1d3e](../../runs/20261016002444872634-811c1d3e/run.json) |  |
| lfm2.5-230m | f16 | cuda | fp8 | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-f16-cuda-fp8-ar) | 60.0 | [20261016005416083372-d372212b](../../runs/20261016005416083372-d372212b/run.json) |  |
| lfm2.5-230m | f16 | webgpu | default | ar | [failed](AKAS.md#lfm2.5-230m-f16-webgpu-default-ar) | 4.5 | — | worker exited with code -9 |
| lfm2.5-230m | f16 | webgpu | fp8 | ar | [budget_exhausted](AKAS.md#lfm2.5-230m-f16-webgpu-fp8-ar) | 60.1 | — | worker wall-clock deadline exceeded |
| lfm2.5-230m | q4_0 | cuda | default | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-q4_0-cuda-default-ar) | 60.0 | [20261016015927998274-7ddf908d](../../runs/20261016015927998274-7ddf908d/run.json) |  |
| lfm2.5-230m | q4_0 | cuda | fp8 | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-q4_0-cuda-fp8-ar) | 60.0 | [20261016022905326483-ef0875a1](../../runs/20261016022905326483-ef0875a1/run.json) |  |
| lfm2.5-230m | q4_0 | webgpu | default | ar | [failed](AKAS.md#lfm2.5-230m-q4_0-webgpu-default-ar) | 4.2 | — | worker exited with code -9 |
| lfm2.5-230m | q4_0 | webgpu | fp8 | ar | [budget_exhausted](AKAS.md#lfm2.5-230m-q4_0-webgpu-fp8-ar) | 60.1 | — | worker wall-clock deadline exceeded |
| lfm2.5-230m | q4_k_m | cuda | default | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-q4_k_m-cuda-default-ar) | 60.0 | [20261016055802779617-648452b2](../../runs/20261016055802779617-648452b2/run.json) |  |
| lfm2.5-230m | q4_k_m | cuda | fp8 | ar | [qualified; budget_exhausted](AKAS.md#lfm2.5-230m-q4_k_m-cuda-fp8-ar) | 60.0 | [20261016040424195157-872d6ba4](../../runs/20261016040424195157-872d6ba4/run.json) |  |
| lfm2.5-230m | q4_k_m | webgpu | default | ar | [failed](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-default-ar) | 4.5 | — | worker exited with code -9 |
| lfm2.5-230m | q4_k_m | webgpu | fp8 | ar | [budget_exhausted](AKAS.md#lfm2.5-230m-q4_k_m-webgpu-fp8-ar) | 60.1 | — | worker wall-clock deadline exceeded |
| qwen3.5-0.8b | bf16 | cuda | default | ar | [budget_exhausted](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-ar) | 60.1 | — | worker exited with code -9; worker wall-clock deadline exceeded |
| qwen3.5-0.8b | bf16 | cuda | default | mtp | [failed](AKAS.md#qwen3.5-0.8b-bf16-cuda-default-mtp) | 1.4 | — | worker exited with code -9 |
| qwen3.5-0.8b | bf16 | cuda | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-ar) | 0.1 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | bf16 | cuda | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-bf16-cuda-fp8-mtp) | 0.0 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | bf16 | webgpu | default | ar | [unsupported](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | bf16 | webgpu | default | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-bf16-webgpu-default-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | bf16 | webgpu | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | bf16 | webgpu | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-bf16-webgpu-fp8-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | f16 | cuda | default | ar | [failed](AKAS.md#qwen3.5-0.8b-f16-cuda-default-ar) | 11.0 | — | GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option; worker exited with code -9 |
| qwen3.5-0.8b | f16 | cuda | default | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-f16-cuda-default-mtp) | 0.1 | — | Qwen GGUF MTP is not implemented; no supported speculative head for this checkpoint/backend |
| qwen3.5-0.8b | f16 | cuda | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-ar) | 0.1 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | f16 | cuda | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-f16-cuda-fp8-mtp) | 0.1 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | f16 | webgpu | default | ar | [unsupported](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | f16 | webgpu | default | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-f16-webgpu-default-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | f16 | webgpu | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | f16 | webgpu | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-f16-webgpu-fp8-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_0 | cuda | default | ar | [failed](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-ar) | 16.9 | — | GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option; worker exited with code -9 |
| qwen3.5-0.8b | q4_0 | cuda | default | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_0-cuda-default-mtp) | 0.1 | — | Qwen GGUF MTP is not implemented; no supported speculative head for this checkpoint/backend |
| qwen3.5-0.8b | q4_0 | cuda | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-ar) | 0.1 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | q4_0 | cuda | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_0-cuda-fp8-mtp) | 0.1 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | q4_0 | webgpu | default | ar | [unsupported](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_0 | webgpu | default | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-default-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_0 | webgpu | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_0 | webgpu | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_0-webgpu-fp8-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_k_m | cuda | default | ar | [failed](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-ar) | 9.5 | — | GGUF head uses fused dequantization followed by argmax; fused_head is a native BF16 option; worker exited with code -9 |
| qwen3.5-0.8b | q4_k_m | cuda | default | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-default-mtp) | 0.1 | — | Qwen GGUF MTP is not implemented; no supported speculative head for this checkpoint/backend |
| qwen3.5-0.8b | q4_k_m | cuda | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-ar) | 0.1 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | q4_k_m | cuda | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_k_m-cuda-fp8-mtp) | 0.1 | — | dense native attention currently requires BF16 KV |
| qwen3.5-0.8b | q4_k_m | webgpu | default | ar | [unsupported](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_k_m | webgpu | default | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-default-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_k_m | webgpu | fp8 | ar | [unsupported](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-ar) | 0.1 | — | native BF16 kernels currently require CUDA |
| qwen3.5-0.8b | q4_k_m | webgpu | fp8 | mtp | [unsupported](AKAS.md#qwen3.5-0.8b-q4_k_m-webgpu-fp8-mtp) | 0.1 | — | native BF16 kernels currently require CUDA |

## Candidate outcomes

169 attempts: 60 budget_exhausted, 83 failed, 26 passed.

| Rejection or deadline | Attempts |
| --- | ---: |
| worker exited with code -9 | 65 |
| worker wall-clock deadline exceeded | 60 |
| ValueError: model logits differ from qualified baseline: b1-p32 | 6 |
| ValueError: model logits differ from qualified baseline: b1-p2048 | 4 |
| ValueError: model logits differ from qualified baseline: b16-p2048 | 2 |
| ValueError: model logits differ from qualified baseline: b64-p2048 | 2 |
| ValueError: model logits differ from qualified baseline: b4-p2048 | 1 |
| ValueError: deployment HTTP output differs from qualified model execution | 1 |
| ValueError: model logits differ from qualified baseline: b4-p32 | 1 |
| ValueError: greedy output differs from baseline for B64/P2048 | 1 |

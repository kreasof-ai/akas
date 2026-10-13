# LFM2-350M performance repair

NVIDIA L40S: CUDA sm_89 and hardware WebGPU/Vulkan. Both safetensors and official F16 GGUF use the composed floating runtime. All 40 configurations and 80 prompt/decode cases completed. Each case uses one excluded warmup, three repetitions, 16 generated steps, capacity 2064 and seed 17. Throughput is the mean, aggregated across requests. No other GPU tests or timing jobs ran concurrently.

The repaired path reuses established arithmetic, chunks prefill, crops the terminal stateless suffix, fuses paired projections and pools disjoint scratch lifetimes. CUDA visits column tiles first to preserve activation reuse at large batches. Both providers share partitioned decode attention. WebGPU chooses GEMV for underfilled grids; a 24-trial beam qualified the 512-workgroup threshold across three geometries and nine row counts. The initial CUDA tile search retained its default. A subsequent 25-trial dispatch beam covered all six projection geometries and 15 row counts. It identified a 32-workgroup floor with a coarse 24-row GEMV ceiling. Full-model boundary checks then caught a 26% drop at batch 25. The final default extends the ceiling through 31 rows and uses a 16-row MMA tile immediately above it, keeping the grid populated. The prepared-plan searches completed after redeployment on 2026-10-13: CUDA covered all six projection geometries and 17 row counts (102 workloads, 12 candidate trials); WebGPU covered three representative geometries and nine row counts (27 workloads, 24 candidate trials). Every measured kernel passed its sampled-row numerical check. Both searches retained the existing defaults: no tested alternative improved the entire workload set without a regression. These bounded searches do not establish an exhaustive optimum for every shape.

Completed validation includes 559 CPU tests (486 skipped), the full performance matrix, long-context diagnostics, dispatch boundaries and shader/IR equivalence. After redeployment, all 87 floating-runtime tests passed across CUDA and hardware WebGPU, along with nine Qwen CUDA regressions and four exact FP8 codec checks. The floating suite covers both loaders, mixed prompt tails, masked decode, partial reset, and partitioned projections around dispatch boundaries. [validation.json](validation.json) records the counts, with JUnit evidence alongside. Installed wheels generated text and benchmarked 32/2048-token prompts with both loaders and both KV formats on both providers while compiler and reference-framework imports were blocked. Within each provider/codec, safetensors and GGUF produced identical generated tokens. The installed `akas bench` CLI also passed compiler-free CUDA and WebGPU checks. [Resumed qualification](resumed-qualification.json) records source revisions, package versions, wheel hashes and search coverage; [CUDA](installed-cuda-resumed.json) and [WebGPU](installed-webgpu-resumed.json) retain the consumer measurements. No engine source changed during this resumed qualification. [Dispatch boundary checks](dispatch-boundaries.json) exercise batches 8/9/24/25/31/32/33 with the same scaling guard. Measured explicit schedules and final defaults are checked for structural TIR equality, per forward requirement, in [default equivalence](cuda-default-equivalence.json). This also verifies unchanged execution for the earlier common-batch measurements. WebGPU banks predating the CUDA-only dispatch update were refreshed and their shader bytes, launch dimensions and argument ABIs proved identical in [shader equivalence](webgpu-shader-equivalence.json).

The full-model sweep stops on numerical errors or a >20% aggregate prefill/decode drop between adjacent batches of the same workload. Per-bank numerical preflight checks distinct three-token prompts in first/last slots. A separate [long-context check](long-context-correctness.json) measures all four provider/codec combinations at batch 1 through a 2048-token prefill and 16 greedy steps. F16 is checked against an independent full trajectory. FP8 long prefill is reported as a drift diagnostic; each FP8 decode check starts from identical quantized GPU state, decoded through independent NumPy equations. The long FP8 prefill does not meet the 2% logit budget, despite matching codec bytes/scales and the first greedy token. This remains a numerical limitation; these performance measurements do not establish language-quality parity.

GEMV uses F32 activations and F16 matrices. GEMM rounds operands to F16 with F32 accumulation. CUDA prefill attention uses F16 operands/probabilities with F32 online sums; WebGPU attention and both decoders use F32. The independent NumPy oracle matches the projection policy, with explicit relative-RMS logit budgets of 0.5% for FP16 KV and 2% for FP8, plus identical greedy tokens in each short preflight. The long diagnostic records token agreement separately. The floating cache ABI quantizes F32 source values directly; the established native BF16 ABI is unchanged. Quantization boundaries amplify small accumulation differences, including changes in near-tied token rankings. WebGPU floating execution requires shader-f16 and 32/64-wide subgroups. Other hardware needs separate qualification.

| Provider | KV | Batch | PP32 | TG32 | PP2048 | TG2048 |
|---|---|---:|---:|---:|---:|---:|
| cuda | float16 | 1 | 9,127 | 761 | 32,451 | 688 |
| cuda | float16 | 4 | 23,866 | 3,073 | 88,972 | 2,707 |
| cuda | float16 | 16 | 31,105 | 6,760 | 116,423 | 5,325 |
| cuda | float16 | 64 | 34,200 | 17,859 | 117,198 | 10,359 |
| cuda | float16 | 256 | 32,001 | 49,844 | 113,209 | 16,968 |
| cuda | fp8 | 1 | 9,120 | 759 | 33,450 | 686 |
| cuda | fp8 | 4 | 23,809 | 3,064 | 89,209 | 2,735 |
| cuda | fp8 | 16 | 30,926 | 6,751 | 116,232 | 5,583 |
| cuda | fp8 | 64 | 34,065 | 17,871 | 119,777 | 11,609 |
| cuda | fp8 | 256 | 31,864 | 49,349 | 112,060 | 21,029 |
| webgpu | float16 | 1 | 3,997 | 470 | 3,368 | 414 |
| webgpu | float16 | 4 | 6,518 | 1,857 | 5,648 | 1,583 |
| webgpu | float16 | 16 | 9,541 | 3,358 | 7,580 | 2,645 |
| webgpu | float16 | 64 | 11,197 | 4,822 | 8,529 | 3,566 |
| webgpu | float16 | 256 | 11,512 | 6,820 | 8,700 | 4,558 |
| webgpu | fp8 | 1 | 3,914 | 476 | 2,910 | 389 |
| webgpu | fp8 | 4 | 6,441 | 1,844 | 5,444 | 1,468 |
| webgpu | fp8 | 16 | 9,564 | 3,356 | 7,474 | 2,474 |
| webgpu | fp8 | 64 | 11,226 | 4,842 | 8,481 | 3,298 |
| webgpu | fp8 | 256 | 11,542 | 6,847 | 8,671 | 4,167 |

This table uses safetensors. [summary.csv](summary.csv) includes all GGUF cases; [measurements.json](measurements.json) records repetitions, preflight errors, generated text and hashes. Compilation, loading and reset are excluded. Older pathological reference measurements used one sample, no warmup and four generated steps, so they give a directional comparison rather than a matched-protocol regression result. The optimized packed CUDA control measured about 39,355 PP2048 tok/s and 705 decode tok/s at batch 1.

Reproduce with the [daily harness](../../../guides/performance-harness.md): use `LFM2Batched`, batches 1/4/16/64/256, prompts 32/2048, generated 16, warmup 1, repeats 3, graphs and `--max-scaling-regression 0.20`. Hardware WebGPU on this adapter requests `--max-buffer-size 2147483647`. The isolated installed consumer blocked compiler/Torch/Transformers imports; the CLI allowed its static profile catalog while blocking Tensor compiler and reference-framework imports. Weights are excluded from the repository; checkpoint hashes are preserved here.

Reproduce the resumed projection searches with the same producer environment:

```bash
python -m benchmarks.kernels.linear --provider cuda \
  --geometries 1024x4608 4608x1024 1024x3072 1024x1024 1024x512 1024x16384 \
  --rows 1 4 8 9 16 24 25 31 32 33 64 128 256 512 2048 8192 32768 \
  --budget 12 --out build/cuda-final-search
python -m benchmarks.kernels.linear --provider webgpu \
  --geometries 4608x1024 1024x3072 1024x1024 \
  --rows 1 4 16 64 256 128 512 2048 8192 \
  --budget 24 --out build/webgpu-final-search
```

The search records both ordinary submission and prepared timing; selection uses the latter with 32 launches per replay. [CUDA search](cuda-final-search/result.json) and [WebGPU search](webgpu-final-search/result.json) retain the results, qualified profiles and all sampled numerical/timing measurements. Search timings use their own kernel protocol and should not be compared directly with earlier ordinary-submission searches or full-model throughput.

# Context-wide Vulkan replay

Selected program SHA256: `96765122afa4a031271fa7f27bd0f74ae6c812c00823901d56dfc834e84ce4ca`.
Checkpoint SHA256: `e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe`.
Run from the AKAS repository in PowerShell:

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m tools.context_sweep akas --checkpoint build/checkpoints/lfm2.5-2.6b/q4_0.gguf --program build/lfm2.5-2.6b-vulkan/tuned-context-v2-final --chunk 512 --prompts benchmarks/runs/20261023155400228961-efa6319e/benchmark/text-workload.json --warmup 0 --repeats 1 --trials 1 --reuse-prefill-check --out build/vulkan-replay-v2.json
```

Rebuild into a new directory with `context_sweep compile`, `--activation-quantization mixed --chunk 512 --prefill-chunks 128 --trials 1`, plus `--schedule-profile benchmarks/runs/20261025121909558322-6c37463a/experiments/final128-profile.json --policy benchmarks/runs/20261025121909558322-6c37463a/experiments/policy-final128.json`. The compiler independently qualifies selected schedules. The retained program links qualified prefill and decode producers with identical weight/state ABIs; their hashes are in `program.json`.

The canonical full report uses runs `20261025121928791585-816cf540` and `20261024191042883833-974e6fff`. The retained native binary is `build/baselines/llama-b11856-vulkan/llama-server.exe`, commit `3642238cc12050c8ab70afdf4b3121ea365c7579`. It uses Vulkan0, 99 GPU layers, context 131072, one slot, batch 2048, ubatch 512, two CPU threads, flash attention and F16 KV. Its retained full sweep has one warmup per prefix and one sample. Queue controls on October 25 use the same binary.

The archived Tensor QAD checkpoint/context-512 forced-token results differ from this standard checkpoint and greedy workload. They informed recipes, not directly matched performance claims.

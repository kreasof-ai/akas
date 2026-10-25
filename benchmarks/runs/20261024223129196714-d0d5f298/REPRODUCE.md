# Vulkan retuning replay

The retained standard Q4_0 checkpoint and token manifest are unchanged.
The full sweep uses one retained sample per prefix, with all compiled phases
loaded before timing. The separate short comparison uses one warmup and three
retained samples for both runtimes. Phase timing boundaries differ by runtime.

## Run the selected compiled program

From the AKAS repository in PowerShell:

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m tools.context_sweep akas --checkpoint build/checkpoints/lfm2.5-2.6b/q4_0.gguf --program build/lfm2.5-2.6b-vulkan/tuned-archive --chunk 512 --prompts benchmarks/runs/20261023155400228961-efa6319e/benchmark/text-workload.json --out build/vulkan-replay.json
```

## Rebuild and requalify the transferred profile

The tuning evidence run 20261024215308077571-967e0923 retains
`diagnostics/selected-archive.json` and `diagnostics/policy-staged.json`.
Pass these to `tools.context_sweep compile --schedule-profile` and `--policy`,
with `--chunk 512 --prefill-chunks 128 --trials 2` and a fresh program directory.
Compilation qualifies and retimes the incumbent, so a future driver can select
different schedules. SCHEDULES.json in the report records what actually ran.

The transferred archive is Tensor d5ee508. Its QAD checkpoint, context-512
forced-token host-logit calls and native b11310 differ from this standard Q4_0
context-131072 greedy workload and native b11856. Its throughput is historical
evidence, rather than a directly matched baseline. This transfer retains F32
decode and F32 prefill totals; archived approximate short-F16 and Q8 activation
prefill modes are not selected.

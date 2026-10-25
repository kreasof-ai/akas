import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akas.benchmarking.history import Recorder, digest, write
from tools.vulkan_plot import plot

root = Path('build/vulkan-tune')
report = Path(Path('build/vulkan-report-final.log').read_text().strip().splitlines()[-1]).parent / 'report.json'
data = json.loads(report.read_text())
assert data['campaign']['status'] == 'completed'
selected = next(p['selected_run_id'] for p in data['profiles'] if p['kind'] == 'akas')
plot(report, root / 'throughput.png')
note = root / 'REPRODUCE.md'
note.write_text('''# Vulkan retuning replay

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
''', encoding='utf-8')
meta = dict(runtime='akas', purpose='derived-figure', protocol='derived-campaign-plot-v1',
    model='LiquidAI/LFM2.5-2.6B', provider='webgpu', target='webgpu-portable-v1',
    weight_format='q4_0', report_sha256=digest(report), input_run_ids=[
        selected, '20261024182401411794-847a6f68', '20261024191042883833-974e6fff'],
    short_comparison_run_ids=['20261024215128041404-5569d5c7', '20261024215220032242-9ad91991'],
    scope='Derived visualization and replay instructions for retained measurements; no new inference samples.')
r = Recorder('benchmarks', meta, kind='preparation', parent=selected,
    change='Plot prior AKAS, retuned AKAS and pinned llama.cpp across all eight prefixes')
for path, name in [(root/'throughput.png', 'throughput.png'),
                   (root/'throughput.svg', 'throughput.svg'),
                   (report, 'input-report.json'),
                   (Path('tools/vulkan_plot.py'), 'plot.py'),
                   (note, 'REPRODUCE.md'), (Path(__file__), 'archive.py')]:
    r.artifact(path, name)
r.update(dict(metadata=meta, results=[]), final=True, status='passed')
write(root/'figure.json', dict(run_id=r.id, status='passed'))
print(r.directory.resolve(), flush=True)

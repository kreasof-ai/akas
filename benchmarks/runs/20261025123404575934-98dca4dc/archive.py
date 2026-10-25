import json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from akas.benchmarking.history import Recorder,digest,write
from tools.campaign_report import measurement
from tools.vulkan_plot import plot
root=Path('build/vulkan-next'); selected='20261025121928791585-816cf540'; baseline='20261024191042883833-974e6fff'; previous='20261024215325218226-fd38f777'; evidence='20261025121909558322-6c37463a'
def run(id):return json.loads(Path('benchmarks/runs',id,'run.json').read_text())
def rows(id,kind):
 r=run(id);return {v['case']['prompt']:(v,measurement(v,kind,True,id,r['metadata']['protocol'])) for v in r['results']}
a,b,old=run(selected),run(baseline),run(previous)
assert all(r['status']=='passed' and r['completed_at'] for r in (a,b,old))
args=[sys.executable,'-m','tools.context_sweep','report','--runs',selected,baseline,'--tuning-run',evidence,'--out',str(root/'report-final')]
result=subprocess.run(args,text=True,capture_output=True,check=True)
(root/'report-final.log').write_text(result.stdout); print(result.stdout,flush=True)
folder=Path(result.stdout.strip().splitlines()[-1]).parent; report=folder/'report.json'
ars,brs,ors=rows(selected,'akas'),rows(baseline,'baseline'),rows(previous,'akas');goals=[]
for prefix,(raw,aa) in sorted(ars.items()):
 bb=brs[prefix][1];oo=ors[prefix][1]
 goals.append(dict(prefix=prefix,prefill=aa['prefill_tok_s'],decode=aa['decode_tok_s'],previous_prefill=oo['prefill_tok_s'],previous_decode=oo['decode_tok_s'],llama_prefill=bb['prefill_tok_s'],llama_decode=bb['decode_tok_s'],prefill_fraction=aa['prefill_tok_s']/bb['prefill_tok_s'],decode_fraction=aa['decode_tok_s']/bb['decode_tok_s'],prefill_target_met=aa['prefill_tok_s']>=bb['prefill_tok_s']*.5,decode_parity_point_met=aa['decode_tok_s']>=bb['decode_tok_s'],same_output_as_previous=raw['output_ids']==ors[prefix][0]['output_ids']))
quality=json.loads((root/'quality-final.json').read_text());assert quality['status']=='passed'
summary=dict(scope='Point comparisons of reported phase rates, not equivalent kernel or end-to-end speedups.',selected_run_id=selected,baseline_run_id=baseline,previous_run_id=previous,prefill_target_fraction=.5,decode_target_fraction=1.,prefill_targets_met=sum(x['prefill_target_met'] for x in goals),decode_targets_met=sum(x['decode_parity_point_met'] for x in goals),all_targets_met=all(x['prefill_target_met'] and x['decode_parity_point_met'] for x in goals),cases=goals)
write(folder/'GOALS.json',summary)
plot(report,root/'throughput.png',prefill_target=.5)
meta=dict(runtime='akas',purpose='derived-figure',protocol='derived-campaign-plot-v1',model='LiquidAI/LFM2.5-2.6B',provider='webgpu',target='webgpu-portable-v1',weight_format='q4_0',report_sha256=digest(report),input_run_ids=[selected,baseline,previous,evidence],scope='Derived plots, target comparisons, output agreements, queue-control discussion, and replay instructions. No new inference samples.')
r=Recorder('benchmarks',meta,kind='preparation',parent=selected,change='Document context-wide Vulkan tuning, measured target misses, and exact replay')
lines=['# RX 6700 XT context-wide Vulkan tuning','',f'The requested targets are **not met at every prefix**. The full sweep clears half of reported llama.cpp prefill at {summary["prefill_targets_met"]}/8 points and reported decode parity at {summary["decode_targets_met"]}/8 points. These are point estimates, with one retained sample per full-sweep prefix.','',f'[Canonical phase comparison](COMPARISON.md) · [Retained target data](GOALS.json) · [Plot](../../runs/{r.id}/throughput.svg)','', 'The same standard LiquidAI LFM2.5-2.6B Q4_0 GGUF, retained C4 token IDs, FP16 KV, B1, 131,072 context capacity, and 128 greedy outputs are used. AKAS runs WebGPU over Vulkan; llama.cpp b11856 uses direct Vulkan. The full AKAS run preloads all phases and has zero per-prefix warmups; the retained pinned llama.cpp full run has one. Fresh one-warmup/three-sample native controls on October 25 agree closely with the retained baseline. Timing boundaries differ, so ratios below evaluate the requested reported-phase targets and do not establish end-to-end speedups.','', '## Prefill','', '| Prefix | Prior AKAS | Selected AKAS | llama.cpp | Fraction of llama | At least half |','| ---: | ---: | ---: | ---: | ---: | :---: |']
for v in goals:lines.append(f'| {v["prefix"]:,} | {v["previous_prefill"]:.1f} | {v["prefill"]:.1f} | {v["llama_prefill"]:.1f} | {v["prefill_fraction"]:.1%} | {"yes" if v["prefill_target_met"] else "no"} |')
lines+=['','## Decode','','| Prefix | Prior AKAS | Selected AKAS | llama.cpp | Fraction of llama | At least parity |','| ---: | ---: | ---: | ---: | ---: | :---: |']
for v in goals:lines.append(f'| {v["prefix"]:,} | {v["previous_decode"]:.1f} | {v["decode"]:.1f} | {v["llama_decode"]:.1f} | {v["decode_fraction"]:.1%} | {"yes" if v["decode_parity_point_met"] else "no"} |')
lines+=['','## Selected changes and qualification','','Prefill uses typed eight-FMA FP16 even/odd partials with FP32 totals for FFNs and triple-width input projections, and two signed-Q8 activation components for other Q4_0 projections. Exact RNE activation staging is timed once before reuse; transient exact F16 weight staging remains timed. This is explicitly different prefill arithmetic from the prior FP32-accumulating program. The Q4_0 checkpoint and external FP16 KV are unchanged. Decode remains FP32 arithmetic over packed shared KV. The selected decoder uses 64-token tiles, 128 physical partitions, overlapping K/V loads, and a subgroup-64 softmax exchange with a smaller-subgroup fallback.','', 'Every selected operation was independently qualified. The final model regression gate matched all 512 teacher-forced predictions across prefixes 32, 512, 2,048 and 8,192, with sampled logit relative RMS below 0.27% (limit 1%) and cosine above 0.9999. This is a bounded regression check against the qualified prior AKAS program, not a corpus accuracy evaluation. The exact-half conversion gate includes ties, signed zero and subnormals. The short-half operation gate combines an analytic rounding bound and a 0.5% RMS limit; the ordinary FP32 gate is unchanged.','',f'The final full-sweep greedy output trajectory matches the prior program at {sum(v["same_output_as_previous"] for v in goals)}/8 prefixes. Cross-runtime output agreement is retained separately in the canonical report. The recurrent-state rewind verification reproduced all 384 prior output IDs in its three-prefix control. It avoids a second prefill and operates outside the phase timers.','', 'Rejected two-pass attention, wider tiles, streamed half loops, direct FP32 dot prototypes, all-half prefill and integer tiling trials are retained in the tuning evidence. Isolated hot-buffer timings and synthetic Gaussian-cache decode screens were used to screen candidates; they are not full-model throughput claims. Some wider packed-query searches were interrupted in compiler layout inference.','', '## Task Manager engines','','The verified RX 6700 XT inventory exposes family 0 with graphics plus compute and family 1 with compute without graphics. The exact installed wgpu-hal 27.0.2 source selects family 0. The pinned llama.cpp source prefers a non-graphics compute family. This matches the observed 3D versus Compute engine labels. Engines can share the same underlying cores ([Microsoft explanation](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/)).','', 'The same llama.cpp binary was measured with `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` and with its default compute queue, each with one warmup and three retained samples. This tests the queue effect in llama.cpp; AKAS itself was not switched to a compute-only queue.','', '| Prefix | Graphics PP | Compute PP | Graphics decode | Compute decode |','| ---: | ---: | ---: | ---: | ---: |']
gfx,cmp=rows('20261025104200757861-c70bd924','baseline'),rows('20261025104227893571-17044c3e','baseline')
for p in sorted(gfx):
 x,y=gfx[p][1],cmp[p][1];lines.append(f'| {p:,} | {x["prefill_tok_s"]:.1f} | {y["prefill_tok_s"]:.1f} | {x["decode_tok_s"]:.1f} | {y["decode_tok_s"]:.1f} |')
lines+=['','The short-context queue effect is small compared with the remaining prefill gap. The inventory, exact crate archive/checksum, source capture, raw queue runs, trial profiles, qualification logs, and model regression captures are retained in tuning evidence `'+evidence+'`.','',f'Full AKAS: `{selected}`. Pinned full llama.cpp: `{baseline}`. Prior AKAS: `{previous}`.','']
(folder/'TUNING.md').write_text('\n'.join(lines),encoding='utf-8')
note=root/'REPRODUCE.md';note.write_text(f'''# Context-wide Vulkan replay

Selected program SHA256: `{digest(Path('build/lfm2.5-2.6b-vulkan/tuned-context-v2-final/program.json'))}`.
Checkpoint SHA256: `e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe`.
Run from the AKAS repository in PowerShell:

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m tools.context_sweep akas --checkpoint build/checkpoints/lfm2.5-2.6b/q4_0.gguf --program build/lfm2.5-2.6b-vulkan/tuned-context-v2-final --chunk 512 --prompts benchmarks/runs/20261023155400228961-efa6319e/benchmark/text-workload.json --warmup 0 --repeats 1 --trials 1 --reuse-prefill-check --out build/vulkan-replay-v2.json
```

Rebuild into a new directory with `context_sweep compile`, `--activation-quantization mixed --chunk 512 --prefill-chunks 128 --trials 1`, plus `--schedule-profile benchmarks/runs/{evidence}/experiments/final128-profile.json --policy benchmarks/runs/{evidence}/experiments/policy-final128.json`. The compiler independently qualifies selected schedules. The retained program links qualified prefill and decode producers with identical weight/state ABIs; their hashes are in `program.json`.

The canonical full report uses runs `{selected}` and `{baseline}`. The retained native binary is `build/baselines/llama-b11856-vulkan/llama-server.exe`, commit `3642238cc12050c8ab70afdf4b3121ea365c7579`. It uses Vulkan0, 99 GPU layers, context 131072, one slot, batch 2048, ubatch 512, two CPU threads, flash attention and F16 KV. Its retained full sweep has one warmup per prefix and one sample. Queue controls on October 25 use the same binary.

The archived Tensor QAD checkpoint/context-512 forced-token results differ from this standard checkpoint and greedy workload. They informed recipes, not directly matched performance claims.
''',encoding='utf-8')
for path,name in [(root/'throughput.png','throughput.png'),(root/'throughput.svg','throughput.svg'),(report,'input-report.json'),(folder/'TUNING.md','TUNING.md'),(folder/'GOALS.json','GOALS.json'),(note,'REPRODUCE.md'),(Path('tools/vulkan_plot.py'),'plot.py'),(Path('tools/context_sweep.py'),'context_sweep.py'),(Path('build/audit_vulkan_reports.py'),'audit_reports.py'),(Path(__file__),'archive.py')]:r.artifact(path,name)
r.update(dict(metadata=meta,results=[]),final=True,status='passed')
write(root/'final-published.json',dict(report=str(folder.resolve()),figure_run_id=r.id,goals=summary))
print('PUBLISHED',folder,r.id,flush=True)

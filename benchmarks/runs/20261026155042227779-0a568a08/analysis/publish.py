"""Seal measured Spark results off main and refresh only its showcase."""
from pathlib import Path
import csv,json,subprocess
from akas.benchmarking.history import Recorder,digest,write
from akas.benchmarking import archive
from tools.campaign_report import record,report_directory
from tools.benchmark_archive import publish,check,prune
from tools.spark_mfu import estimate
from akas.models.spark import SparkCheckpoint
h=Path('benchmarks');w=Path('build/spark-parity');py='.venv/bin/python'
ids=[json.loads((w/f'{r}-full.json').read_text())['run_id'] for r in ('akas','llama')]
records=[record(h,i) for i in ids]
assert all(r['status']=='passed' and not r['source']['changed_during_run'] for r in records)
assert all([x['case']['prompt'] for x in r['results']]==[32,512,2048,8192,16384,32768,65536,130944] for r in records)
subprocess.run([py,'-m','tools.spark_benchmark','report','--runs',*ids,'--out',str(w/'report')],check=True)
state=json.loads((w/'report/campaign.json').read_text());directory=report_directory(state,h);rid=directory.name
data=json.loads((directory/'report.json').read_text());a,b=data['profiles']
c=SparkCheckpoint('build/checkpoints/spark-x2.6-4b/gguf/Spark-X2.6-4B-Q4_K_M.gguf',tokenizer=False).config
mfu=estimate(data,c);write(w/'mfu.json',mfu);mfus={r['prefix']:r['mfu_percent'] for r in mfu['rows'] if r['runtime']=='akas'}
def peak(runtime):
 rows=list(csv.DictReader((w/f'gpu-{runtime}.csv').open()))
 key=next(k for k in rows[0] if k.strip().startswith('memory.used'))
 return max(int(r[key].strip().split()[0]) for r in rows)
peaks={k:peak(k) for k in ('akas','llama')};rows=[]
for left,right in zip(a['measurements'],b['measurements']):
 n=left['prefix']
 rows.append(dict(prefix=n,prefill=left['prefill_tok_s'],llama_prefill=right['prefill_tok_s'],
  decode=left['decode_tok_s'],llama_decode=right['decode_tok_s'],
  prefill_ratio=left['prefill_tok_s']/right['prefill_tok_s'],decode_ratio=left['decode_tok_s']/right['decode_tok_s'],
  all_decode_samples_ahead=min(s['decode_tokens_per_second'] for s in left['samples'])>max(s['decode_tokens_per_second'] for s in right['samples']),mfu_percent=mfus[n]))
summary=dict(rows=rows,peak_gpu_MiB=peaks,source_runs=ids,source_commit=records[0]['source']['revision'],
 allocated_weight_bytes=records[0]['metadata']['allocated_weight_bytes'],mapped_host_weight_bytes=records[0]['metadata']['mapped_host_weight_bytes'],
 allocated_state_bytes=records[0]['metadata']['allocated_state_bytes'],persistent_dequantized_weight_bytes=0,
 prefill_medians_ahead=sum(r['prefill_ratio']>=1 for r in rows),decode_medians_ahead=sum(r['decode_ratio']>1 for r in rows),
 decode_ranges_ahead=sum(r['all_decode_samples_ahead'] for r in rows),baseline_reused=True)
write(w/'assessment.json',summary)
lines=['# Spark X2.6: compact packed-weight prefill','',
 'Both runtimes were measured in this optimization session on the same NVIDIA L4 with the pinned official Q4_K_M checkpoint and identical synthetic input IDs. One request, 128 total output tokens, empty cache per request, FP16 KV, one warmup and three retained repetitions. Full 131072-token context reservation. Loading, compilation and phase preparation are excluded. Native synchronized AKAS phases and llama-server eval phases have different timing boundaries.','',
 f"Observed peak GPU memory (nvidia-smi total used): **AKAS {peaks['akas']/1024:.2f} GiB ({peaks['akas']:,} MiB); llama.cpp {peaks['llama']/1024:.2f} GiB ({peaks['llama']:,} MiB)**. The earlier packed-weight AKAS run peaked at 9.50 GiB. GPU weight storage including compact rotary tables: {summary['allocated_weight_bytes']/2**30:.3f} GiB. Persistent state: {summary['allocated_state_bytes']/2**30:.3f} GiB, almost entirely FP16 KV.",'',
 f"Both runtimes keep the input embedding on the CPU. AKAS maps {summary['mapped_host_weight_bytes']/2**20:.0f} MiB of pinned host memory; all transformer weights and the output head stay on the GPU. Quantized weights retain checkpoint bytes in aligned storage. There is no persistent expanded FP16 weight matrix. Temporary on-chip tiles supply tensor-core operands.",'',
 'AKAS uses a 2048-token maximum prefill chunk; llama.cpp uses 512. Both have the same full-attention capacity and KV precision. AKAS retains one sliding window per layer and reuses a temporary current-chunk K/V workspace across layers, committing only the live suffix after attention. Shared state allocations avoid per-layer allocation padding. Cross-phase scratch and identical kernel modules are shared. Compact rotary tables replace full-context tables. Fused prefill combines gate/up/exact GELU, residual addition/normalization, and QKV split/rotary/cache writes. Activation staging uses the following MMA operation’s FP16 precision; reductions and residuals stay FP32.','',
 f"Prefill medians meet or exceed llama.cpp at {summary['prefill_medians_ahead']}/8 prefixes. Decode medians exceed it at {summary['decode_medians_ahead']}/8; all retained decode samples exceed corresponding baseline samples at {summary['decode_ranges_ahead']}/8. These are measurements of this configuration and workload, not a guarantee for arbitrary inputs or hardware.",'',
 '| Prefix | AKAS prefill tok/s | llama.cpp prefill tok/s | AKAS decode tok/s | llama.cpp decode tok/s | Estimated prefill MFU |',
 '| ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:lines.append(f"| {r['prefix']:,} | {r['prefill']:,.1f} | {r['llama_prefill']:,.1f} | {r['decode']:,.2f} | {r['llama_decode']:,.2f} | {r['mfu_percent']:.2f}% |")
lines+=['',
 'Useful prefill MFU counts dense projections, one final-row vocabulary projection and causal QK/PV work against [NVIDIA L4’s 121 TFLOP/s dense FP16 peak](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf). It excludes dequantization, padding, elementwise operations, embedding lookups and cache/control traffic. It counts all model layers even where intermediate chunks prune final-layer outputs. It is not measured hardware instruction utilization.','',
 'An incomplete development run (`20261026145843633286-516a8e7e`) was rejected after a pruning bug was discovered. Its sealed record is retained for audit and must not be used for comparisons. The current run uses mapped-output identities preserved by name and a four-layer oracle covering middle full attention.','', '## Qualification','',
 'Independent operation equations cover packed Q4_K, Q6_K, Q8_0 and BF16 storage, odd matrix tails, FP16 activation staging, exact GELU, mixed-precision residual RMS, high-position compact RoPE, fused QKV writes, inactive rows, wrap and NaN padding. Complete BF16, native channel-FP8 and four-layer mixed-GGUF models are compared with independent Torch equations across chunk boundaries, physical slot reorder and reuse. GPU uploads are checked byte for byte. Allocation tests cover shared scratch, shared module integrity, typed arena slices, mapped embeddings and ownership after another phase closes. The corrected 4B checkpoint also produces bitwise-identical 131072 logits at an 8192-token prompt with intermediate state prefill enabled or disabled. Two legacy mapping tests reference removed entry points and fail identically on the pre-fix checkout; their logs are retained as exclusions. These tests do not establish full pretrained-model task accuracy. Cross-runtime greedy trajectories are retained separately in the canonical report. GPU measurements are serialized under stock clock/power management; the retained telemetry includes temperature, clocks and power. The fresh llama.cpp run was collected earlier in this session with the same pinned binary, checkpoint and protocol, and reused after the native compact-cache refinement.','',
 '## Reproduce','', '```sh',
 'python -m tools.spark_benchmark compile --optimized --chunk 2048 --program build/spark/compact --out build/spark/compile.json',
 'python -m tools.spark_benchmark akas --chunk 2048 --program build/spark/compact --out build/spark/akas.json',
 'python -m tools.spark_benchmark llama --out build/spark/llama.json',
 'python -m tools.spark_benchmark report --runs AKAS_RUN LLAMA_RUN --out build/spark/report',
 'python -m tools.spark_mfu REPORT_DIRECTORY/report.json --checkpoint MODEL.gguf --out build/spark/mfu.json','```','']
(w/'assessment.md').write_text('\n'.join(lines))
subprocess.run([py,str(w/'plot.py'),str(directory/'report.json'),str(w/'throughput')],check=True)
meta=dict(runtime='akas',purpose='benchmark-figure',model='XHToken/Spark-X2.6-4B',mode='ar',weight_format='q4_k_m',provider='cuda',target='sm_89',kv_dtype='float16',source_runs=ids,report_id=rid,report_sha256=digest(directory/'report.json'),persistent_dequantized_weight_bytes=0,baseline_reused=True)
r=Recorder(h,meta,kind='verification',change='Retain compact packed Spark throughput, MFU and matched memory assessment')
for file,name in [('throughput.svg','figures/throughput.svg'),('throughput.png','figures/throughput.png'),('plot.py','figures/plot.py'),('mfu.json','analysis/mfu.json'),('assessment.json','analysis/assessment.json'),('assessment.md','analysis/README.md'),('publish.py','analysis/publish.py'),('gpu-akas.csv','environment/gpu-akas.csv'),('gpu-llama.csv','environment/gpu-llama.csv'),('benchmark.sh','environment/benchmark.sh'),('compile-compact.py','environment/compile.py'),('gpu-akas-initial.csv','environment/gpu-akas-initial.csv'),('gpu-akas-retracted.csv','environment/gpu-akas-retracted.csv')]:r.artifact(w/file,name)
r.artifact('tools/spark_mfu.py','analysis/spark_mfu.py')
x=r.update(dict(schema='akas.benchmark.v1',metadata=meta,results=[]),final=True,status='passed');assert x['status']=='passed';figure_id=r.id
before=archive.catalog(h);after=publish(h);check(h,objects=True,tracked=True)
changed=sorted({entry['branch'] for kind in ('runs','reports') for ident,entry in after[kind].items() if ident not in before[kind]})
subprocess.run(['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential','push','origin',*changed],check=True)
retraction_id=json.loads((w/'retraction.json').read_text())['run_id'];retraction_url=archive.link(h,'runs',retraction_id,'analysis/README.md');
report_url=archive.link(h,'reports',rid,'README.md');assessment_url=archive.link(h,'runs',figure_id,'analysis/README.md');figure_url=archive.link(h,'runs',figure_id,'figures/throughput.svg',raw=True)
lines=['**Spark X2.6-4B · Q4_K_M weights · NVIDIA L4 (23,034 MiB) · CUDA**','',
 f'Results from the [October 26, 2026 compact packed-weight comparison]({report_url}).',
 'Both runtimes were measured with one request generating 128 total output tokens,',
 'identical official checkpoint bytes, synthetic prompt IDs and FP16 KV. Rates',
 'are medians of three retained repetitions after one warmup. Quantized weights',
 'stay packed; no persistent FP16 weight copy is allocated.','',
 f'![Spark X2.6 Q4_K_M prefill and decode throughput with observed sample ranges.]({figure_url})','',
 '<details>','<summary>Measurements (tokens per second) and estimated prefill MFU</summary>','',
 '| Prefix tokens | AKAS prefill tok/s | llama.cpp prefill tok/s | AKAS decode tok/s | llama.cpp decode tok/s | Estimated AKAS prefill MFU |',
 '| ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:lines.append(f"| {r['prefix']:,} | {r['prefill']:,.1f} | {r['llama_prefill']:,.1f} | {r['decode']:,.1f} | {r['llama_decode']:,.1f} | {r['mfu_percent']:.2f}% |")
lines+=['','</details>','',
 f"Peak GPU memory was **{peaks['akas']/1024:.2f} GiB for AKAS and {peaks['llama']/1024:.2f} GiB for llama.cpp**",
 'with the full 131072-token context reservation. Both keep the input embedding',
 'on the CPU; AKAS maps 640 MiB of pinned host memory. All transformer weights',
 'and the output head remain on the GPU. The earlier packed-weight AKAS run',
 'peaked at 9.50 GiB. Shared allocations and scratch, compact rotary tables,',
 'fused prefill, compact sliding rings and FP16 activation staging reduce memory and prefill traffic.','',
 f"Prefill medians meet or exceed llama.cpp at {summary['prefill_medians_ahead']}/8 prefixes. Decode medians",
 f"exceed it at {summary['decode_medians_ahead']}/8, with all retained decode samples ahead at {summary['decode_ranges_ahead']}/8.",
 'These observations describe this measured configuration. AKAS uses a maximum',
 '2048-token prefill chunk and llama.cpp uses 512. Native synchronized AKAS',
 'phases and llama-server eval phases have different timing boundaries; loading',
 'is excluded. The sealed llama.cpp run from earlier in this session is reused',
 'with its pinned binary and matching workload unchanged. Estimated MFU counts',
 'useful model math against L4’s 121 TFLOP/s',
 'dense FP16 peak and is not a hardware instruction counter.','',
 '[Spark model support and usage](src/akas/models/spark/README.md) also cover BF16',
 'safetensors and native FP8 weights with channel scales. See the',
 f'[memory assessment, MFU method and GPU telemetry]({assessment_url})',
 f'and [full report]({report_url}) for raw samples, output comparisons and selected',
 'schedules. Earlier results remain immutable in [the archive catalog](benchmarks/archive.json).','']
guide=Path('benchmarks/README.md');body=guide.read_text();body+=f'\nA development run (`20261026145843633286-516a8e7e`) was rejected after a compiler pruning error was found. [Correction and rejected-run evidence]({retraction_url}). Its original record is retained for audit and must not seed performance comparisons. The showcase uses the corrected, four-layer-validated program.\n';guide.write_text(body);
readme=Path('README.md');content=readme.read_text();start=content.index('**Spark X2.6-4B');end=content.index('## Getting started',start);readme.write_text(content[:start]+'\n'.join(lines)+'\n'+content[end:])
prune(h);verified=archive.directory(h,'reports',rid,fetch=True)
subprocess.run([py,'-m','tools.campaign_report','--check'],check=True);check(h,objects=True,tracked=True)
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add','README.md','benchmarks/README.md','benchmarks/archive.json','benchmarks/index.json','benchmarks/RESULTS.md','benchmarks/ARCHIVE.md','benchmarks/reports/README.md'],check=True)
subprocess.run(['git','commit','-m','Archive matched-memory Spark benchmarks and refresh showcase'],check=True)
summary.update(report_id=rid,report_url=report_url,assessment_url=assessment_url,figure_run_id=figure_id,figure_url=figure_url,artifact_branches=changed,report_directory=str(verified),publication_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
write(w/'completion.json',summary);print(json.dumps(summary,indent=2),flush=True)

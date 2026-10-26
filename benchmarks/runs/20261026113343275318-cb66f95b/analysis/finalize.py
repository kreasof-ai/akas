"""Archive packed-only Spark measurements and replace the superseded showcase."""
from pathlib import Path
import csv,json,subprocess
from akas.benchmarking.history import Recorder,digest,write
from akas.benchmarking import archive
from tools.campaign_report import record,report_directory
from tools.benchmark_archive import publish,check,prune
from tools.spark_mfu import estimate
from akas.models.spark import SparkCheckpoint
h=Path('benchmarks');w=Path('build/spark-packed');py='.venv/bin/python'
native=json.loads((w/'akas-full.json').read_text())['run_id'];baseline='20261026101811969664-715fa9e0';ids=[native,baseline];records=[record(h,i) for i in ids]
assert all(r['status']=='passed' and not r['source']['changed_during_run'] for r in records)
assert all([x['case']['prompt'] for x in r['results']]==[32,512,2048,8192,16384,32768,65536,130944] for r in records)
subprocess.run([py,'-m','tools.spark_benchmark','report','--runs',*ids,'--out',str(w/'report')],check=True)
state=json.loads((w/'report/campaign.json').read_text());directory=report_directory(state,h);rid=directory.name;data=json.loads((directory/'report.json').read_text());a,b=data['profiles']
c=SparkCheckpoint('build/checkpoints/spark-x2.6-4b/gguf/Spark-X2.6-4B-Q4_K_M.gguf',tokenizer=False).config
mfu=estimate(data,c);write(w/'mfu.json',mfu);mfus={r['prefix']:r['mfu_percent'] for r in mfu['rows'] if r['runtime']=='akas'}
telemetry=list(csv.DictReader((w/'gpu-timeseries.csv').open()));peak=max(int(r[' memory.used [MiB]'].strip().split()[0]) for r in telemetry);rows=[]
for left,right in zip(a['measurements'],b['measurements']):
 n=left['prefix'];nmin=min(s['decode_tokens_per_second'] for s in left['samples']);bmax=max(s['decode_tokens_per_second'] for s in right['samples'])
 rows.append(dict(prefix=n,prefill=left['prefill_tok_s'],llama_prefill=right['prefill_tok_s'],decode=left['decode_tok_s'],llama_decode=right['decode_tok_s'],prefill_ratio=left['prefill_tok_s']/right['prefill_tok_s'],decode_ratio=left['decode_tok_s']/right['decode_tok_s'],all_decode_samples_ahead=nmin>bmax,mfu_percent=mfus[n]))
summary=dict(rows=rows,peak_gpu_MiB=peak,peak_gpu_GiB=peak/1024,allocated_weight_bytes=records[0]['metadata']['allocated_weight_bytes'],allocated_state_bytes=records[0]['metadata']['allocated_state_bytes'],persistent_dequantized_weight_bytes=0,baseline_reused=True,source_runs=ids,decode_medians_ahead=sum(r['decode_ratio']>1 for r in rows),decode_ranges_ahead=sum(r['all_decode_samples_ahead'] for r in rows),source_commit=records[0]['source']['revision'])
write(w/'assessment.json',summary)
lines=['# Spark X2.6 with packed quantized weights','',
 'This run removes the persistent FP16 prefill weight copy. GGUF matrices remain in their checkpoint encodings for both prefill and decode. FP8 safetensors now retain E4M3 bytes and BF16 channel scales. Quantized coefficients are decoded only in temporary compute tiles; no full persistent FP16 matrix is created. The removed prefill_weight_cache option is rejected.','',
 f"Peak observed GPU memory: **{peak/1024:.2f} GiB ({peak:,} MiB)**, compared with 16.63 GiB in the superseded cached-weight run. Resident weight allocations including rotary tables: {summary['allocated_weight_bytes']/2**30:.2f} GiB. Persistent state: {summary['allocated_state_bytes']/2**30:.2f} GiB, almost entirely FP16 KV. The remaining memory covers workspaces, loaded kernels, graphs and CUDA overhead.",'',
 'One request, prefixes 32 through 130944, 128 total output tokens, FP16 KV, one warmup and three retained samples per prefix. Loading and phase preparation are outside the timers. Native AKAS phase timing and llama.cpp server eval timing have different boundaries.','',
 f"The comparison reuses sealed llama.cpp run `{baseline}` from earlier on October 26, 2026. Its checkpoint bytes, prompt IDs, GPU, output count, KV dtype, warmup/repetition counts and pinned binary are unchanged. llama.cpp was not rerun for this implementation-only change.",'',
 f"AKAS decode medians exceed llama.cpp at {summary['decode_medians_ahead']}/8 prefixes, and all retained decode samples exceed corresponding llama.cpp samples at {summary['decode_ranges_ahead']}/8. These results describe the measured workload, rather than guaranteeing all inputs or hardware.",'',
 '| Prefix | AKAS prefill tok/s | llama.cpp prefill tok/s | AKAS decode tok/s | llama.cpp decode tok/s | Estimated prefill MFU |','| ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:lines.append(f"| {r['prefix']:,} | {r['prefill']:,.1f} | {r['llama_prefill']:,.1f} | {r['decode']:,.2f} | {r['llama_decode']:,.2f} | {r['mfu_percent']:.2f}% |")
lines+=['','Prefill MFU estimates useful dense projections, one final-row vocabulary projection and causal QK/PV work, relative to [NVIDIA L4’s 121 TFLOP/s dense FP16 peak](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf). It excludes embedding lookups, elementwise work, padding, dequantization and cache/control traffic; it counts all layers even where intermediate chunks prune final-layer outputs. It is not measured hardware instruction utilization.','',
 '## Qualification','',
 'Storage tests verify original GGUF/FP8 bytes are preserved and no expanded quantized GPU weight matrices are allocated. Independent CUDA operation checks and BF16/FP8 small-model Torch references cover cache wrap, reorder, slot reuse, inactive requests and NaN padding. The actual 4B FP8 checkpoint also passes a finite-logit decode smoke check with native FP8 storage. The long-reduction FP8 projection matches the existing F16-operand/F32-accumulation GEMM bit for bit on all 81,920 tested outputs; its absolute qualification tolerance matches packed projections. Supplemental logs and generators are retained in the qualification run. Full-model cross-runtime greedy trajectory comparisons remain separate from timing in the canonical report.','',
 '## Reproduce','', '```sh',
 'python -m tools.spark_benchmark compile --optimized --chunk 1024 --program build/spark/packed --out build/spark/packed-compile.json',
 'python -m tools.spark_benchmark akas --chunk 1024 --program build/spark/packed --out build/spark/packed-run.json',
 'python -m tools.spark_benchmark report --runs AKAS_RUN 20261026101811969664-715fa9e0 --out build/spark/packed-report',
 'python -m tools.spark_mfu REPORT_DIRECTORY/report.json --checkpoint MODEL.gguf --out build/spark/mfu.json','```','',
 'The optimized option enables shared rotary tables and optimized attention. Quantized weights remain packed. Recompile older programs that enabled the removed FP16 prefill cache or used the former FP8-to-FP16 import layout.','']
(w/'assessment.md').write_text('\n'.join(lines))
subprocess.run([py,str(w/'plot.py'),str(directory/'report.json'),str(w/'throughput')],check=True)
meta=dict(runtime='akas',purpose='benchmark-figure',model='XHToken/Spark-X2.6-4B',mode='ar',weight_format='q4_k_m',provider='cuda',target='sm_89',kv_dtype='float16',source_runs=ids,report_id=rid,report_sha256=digest(directory/'report.json'),persistent_dequantized_weight_bytes=0,baseline_reused=True,scope='Packed-only quantized weights: throughput, useful MFU and GPU memory telemetry')
r=Recorder(h,meta,kind='verification',change='Retain Spark packed-weight throughput and memory assessment')
for file,name in [('throughput.svg','figures/throughput.svg'),('throughput.png','figures/throughput.png'),('plot.py','figures/plot.py'),('mfu.json','analysis/mfu.json'),('assessment.json','analysis/assessment.json'),('assessment.md','analysis/README.md'),('finalize.py','analysis/finalize.py'),('gpu-timeseries.csv','environment/gpu-timeseries.csv')]:r.artifact(w/file,name)
r.artifact('tools/spark_mfu.py','analysis/spark_mfu.py')
x=r.update(dict(schema='akas.benchmark.v1',metadata=meta,results=[]),final=True,status='passed');assert x['status']=='passed';figure_id=r.id
before=json.loads((h/'archive.json').read_text());after=publish(h);check(h,objects=True,tracked=True)
changed=sorted({entry['branch'] for kind in ('runs','reports') for ident,entry in after[kind].items() if ident not in before[kind]})
subprocess.run(['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential','push','origin',*changed],check=True)
report_url=archive.link(h,'reports',rid,'README.md');assessment_url=archive.link(h,'runs',figure_id,'analysis/README.md');entry=after['runs'][figure_id];url=f"https://raw.githubusercontent.com/kreasof-ai/akas/{entry['commit']}/{entry['path']}/figures/throughput.svg"
lines=['**Spark X2.6-4B · Q4_K_M weights · NVIDIA L4 (23,034 MiB) · CUDA**','',
 f'Results from the [October 26, 2026 packed-weight comparison]({report_url}),',
 'with one request generating 128 total output tokens. Quantized weights remain',
 'packed on the GPU for both prefill and decode, with no persistent FP16 weight',
 'copy. Both runtimes use the same official checkpoint, synthetic prompt IDs and',
 'FP16 KV. Rates are medians of three retained repetitions after one warmup.', '',
 f'![Spark X2.6 Q4_K_M prefill and decode throughput with packed weights and observed sample ranges.]({url})','',
 '<details>','<summary>Measurements (tokens per second) and estimated prefill MFU</summary>','',
 '| Prefix tokens | AKAS prefill tok/s | llama.cpp prefill tok/s | AKAS decode tok/s | llama.cpp decode tok/s | Estimated AKAS prefill MFU |',
 '| ---: | ---: | ---: | ---: | ---: | ---: |']
for row in rows:lines.append(f"| {row['prefix']:,} | {row['prefill']:,.1f} | {row['llama_prefill']:,.1f} | {row['decode']:,.1f} | {row['llama_decode']:,.1f} | {row['mfu_percent']:.2f}% |")
lines+=['','</details>','',
 f"Peak GPU memory was **{peak/1024:.2f} GiB**, including the full 131072-token KV",
 'reservation for one request. The former FP16 prefill weight copy has been removed.',
 f"Decode medians exceed llama.cpp at {summary['decode_medians_ahead']}/8 prefixes; all retained",
 f"decode samples exceed corresponding llama.cpp samples at {summary['decode_ranges_ahead']}/8. These",
 'observations describe this measured configuration. The chart reuses the sealed',
 'llama.cpp baseline from earlier on October 26, with identical checkpoint bytes,',
 'prompts, GPU, KV precision and sampling protocol; llama.cpp was not rerun.', '',
 'AKAS uses synchronized native phases; llama.cpp uses server eval phases.',
 'Loading is excluded and phase timing boundaries differ. Estimated MFU counts',
 'useful model math relative to L4’s 121 TFLOP/s dense FP16 peak; it is not a',
 'hardware instruction counter. The optimized configuration uses shared rotary',
 'tables, optimized attention and a 1024-token maximum prefill chunk.', '',
 '[Spark model support and usage](src/akas/models/spark/README.md) also cover BF16',
 'safetensors and native FP8 weight storage with channel scales. See the',
 f'[memory assessment, MFU method and GPU telemetry]({assessment_url})',
 f'and the [full report]({report_url}) for raw samples, output comparisons and',
 'selected schedules. Earlier results using the removed FP16 weight copy remain',
 'historical records in [the archive catalog](benchmarks/archive.json).','']
readme=Path('README.md');content=readme.read_text();start=content.index('**Spark X2.6-4B');end=content.index('## Getting started',start);readme.write_text(content[:start]+'\n'.join(lines)+'\n'+content[end:])
prune(h);verified=archive.directory(h,'reports',rid,fetch=True)
subprocess.run([py,'-m','tools.campaign_report','--check'],check=True);check(h,objects=True,tracked=True)
subprocess.run(['git','diff','--check'],check=True)
subprocess.run(['git','add','README.md','benchmarks/archive.json','benchmarks/index.json','benchmarks/RESULTS.md','benchmarks/ARCHIVE.md','benchmarks/reports/README.md'],check=True)
subprocess.run(['git','commit','-m','Archive packed-weight Spark benchmark and replace cached-weight showcase'],check=True)
summary.update(report_id=rid,report_url=report_url,assessment_url=assessment_url,figure_run_id=figure_id,figure_url=url,artifact_branches=changed,report_directory=str(verified),publication_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
write(w/'completion.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)

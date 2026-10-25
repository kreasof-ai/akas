"""Record kernel latency experiments without presenting them as model throughput."""
import json,resource,subprocess,sys,traceback
from pathlib import Path
from akas.benchmarking.history import Recorder,write
from tensor.artifacts.format import read_artifact
root=Path('build/lfm-moe/kernel-pilots');root.mkdir(exist_ok=True)
for provider,stages in (
 ('cuda',[('pipeline','build/lfm-moe/attention-cuda-async.py',[]),('experts','build/lfm-moe/moe-decode-explore.py',[])]),
 ('webgpu',[('prefill','build/lfm-moe/attention-webgpu-reuse.py',[]),('decode','build/lfm-moe/attention-decode-explore.py',['webgpu'])]),
):
 meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider=provider,target='sm_89' if provider=='cuda' else 'webgpu-portable-v1',device=dict(name='NVIDIA L4',provider=provider),definition='akas.models.lfm2.model:LFM2MoE',weights='synthetic encoded/float cache samples; no full-model rates',workload=dict(query_rows=512,heads=32,kv_heads=8,head_dim=64,kv_dtype='float16',prefill_context=8192,decode_contexts=[32,8192,131072]),experiments=[])
 recorder=Recorder('benchmarks',meta,kind='algorithm',change='Independently qualify and time attention load pipelining, KV-vector reuse and grouped decode; kernel latencies only')
 report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
 try:
  for label,runner,args in stages:
   log=root/f'{provider}-{label}.log'
   with log.open('w') as stream:subprocess.run([sys.executable,'-u',runner,*args],stdout=stream,stderr=subprocess.STDOUT,check=True)
   rows=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
   assert rows and all(r.get('correct') is True for r in rows)
   meta['experiments'].append(dict(stage=label,measurements=rows))
   recorder.artifact(log,f'kernels/{label}.log');recorder.artifact(runner,f'kernels/{label}-runner.py')
  artifacts=([Path('build/lfm-moe/attention-cuda-async')/f'8192-{name}.tbin' for name in ('serial','async2','async3','asyncq64')] + list(Path('build/lfm-moe/moe-decode-explore').glob('*.tbin')) if provider=='cuda' else [Path('build/lfm-moe/attention-tiled_prefill.tbin'),Path('build/lfm-moe/attention-reuse.tbin'),*Path('build/lfm-moe/attention-decode-explore/webgpu').glob('*.tbin')])
  meta["kernel_artifacts"]=[dict(label=p.name,path=f"kernels/{i}/manifest.json") for i,p in enumerate(artifacts)]
  for i,p in enumerate(artifacts):
   manifest,files=read_artifact(p)
   out=root/provider/str(i);out.mkdir(parents=True,exist_ok=True)
   write(out/'manifest.json',manifest);recorder.artifact(out/'manifest.json',f'kernels/{i}/manifest.json')
   for name,content in files.items():
    if name.endswith(('.tirx.json','.wgsl')):
     (out/name).write_bytes(content);recorder.artifact(out/name,f'kernels/{i}/{name}')
  passed=True
 except BaseException as e:
  meta['error']=f'{type(e).__name__}: {e}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
 finally:
  meta['child_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
  recorder.artifact(__file__,'kernels/runner.py');recorder.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',provider,recorder.id,passed,flush=True)
 if not passed:sys.exit(1)

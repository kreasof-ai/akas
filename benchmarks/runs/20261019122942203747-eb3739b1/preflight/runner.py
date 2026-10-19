"""Retain compilation resources, correctness and capacity evidence together."""
import json
from pathlib import Path
import subprocess
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from akas.benchmarking.history import Recorder, write, utc

root=Path('build/lfm-moe')
python=sys.executable
subprocess.run([python, '-c', 'import tensor; import akas; print("Worker interpreter verified")'],check=True)
checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
sha='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5'
def step(provider,stage,command):
 write(root/'preflight-progress.json',dict(provider=provider,stage=stage,updated_at=utc()))
 print(provider,stage,utc(),flush=True)
 with (root/f'{provider}-{stage}.log').open('w') as log:
  subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
for provider in ('cuda','webgpu'):
 target='sm_89' if provider=='cuda' else 'webgpu-portable-v1'
 meta=dict(runtime='akas',purpose='preflight',definition='akas.models.lfm2.model:LFM2MoE',checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=sha,provider=provider,target=target,device=dict(name='NVIDIA L4',provider=provider),protocol='native-token-step-v2',weight_format='q4_0',mode='ar',kv_dtype='float16')
 recorder=Recorder('benchmarks',meta,kind='validation',change='Qualify compact expert tiles, tiled WebGPU prefill and bounded-memory search at full model context')
 report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
 compiled=False
 passed=False
 executed=set()
 try:
  executed.add('seed-search')
  step(provider,'seed-search',[python,'-u',str(root/'seed-search.py'),provider])
  executed.add('adaptive-qualification')
  step(provider,'adaptive-qualification',[python,'-u',str(root/'qualify-adaptive.py'),provider])
  compiled=True
  buffer=['--max-buffer-size','268435456'] if provider=='webgpu' else []
  executed.add('text-sanity')
  step(provider,'text-sanity',[python,'-m','akas','generate','--definition',meta['definition'],'--model',checkpoint,'--bundle',str(root/f'{provider}-512'),'--provider',provider,*buffer,'--prompt','What is the capital of France? Answer briefly.','--max-tokens','128','--out',str(root/f'{provider}-text-sanity.json')])
  assert 'paris' in json.loads((root/f'{provider}-text-sanity.json').read_text())['text'].lower()
  executed.add('capacity-pilot')
  (root/f'{provider}-capacity-pilot.json').unlink(missing_ok=True)
  step(provider,'capacity-pilot',[python,'-m','akas','bench','--no-history','--definition',meta['definition'],'--checkpoint',checkpoint,'--checkpoint-options','{"tokenizer":false}','--bundles',str(root/'bench-bank-512'),'--provider',provider,'--target',target,'--kv-dtype','float16',*buffer,'--batches','1','--prompts','32','8192','130944','--generated','128','--prefill-chunk','512','--repeats','1','--warmup','0','--graphs','--out',str(root/f'{provider}-capacity-pilot.json')])
  report=json.loads((root/f'{provider}-capacity-pilot.json').read_text())
  assert len(report['results'])==3 and all(row['status']=='ok' for row in report['results'])
  report['metadata'].update({key:value for key,value in meta.items() if key != 'device'})
  report['metadata']['run_id']=recorder.id
  report['metadata']['qualification']=dict(operations='independent numerical references for every selected kernel',text='real checkpoint generation contains Paris; not a whole-model long-context oracle',capacity='131072 total tokens, FP16 KV, no CPU offload')
  resources=[json.loads(line.split('RESOURCE ',1)[1]) for line in (root/f'{provider}-adaptive-qualification.log').read_text().splitlines() if line.startswith('RESOURCE ')]
  assert len(resources)==1
  report['metadata']['compilation_resources']=resources[0]
  seeds=[json.loads(line) for line in (root/f'{provider}-seed-search.log').read_text().splitlines() if line.startswith('{"provider"')]
  report['metadata']['kernel_seed_search_resources']=seeds
  write(root/f'{provider}-capacity-pilot.json',report)
  passed=True
 except BaseException as error:
  report['metadata']['preflight_error']=f'{type(error).__name__}: {error}'
  report['metadata']['preflight_traceback']=traceback.format_exc()
  raise
 finally:
  if compiled:
   recorder.program(root/f'{provider}-512')
  for stage in executed:
   path=root/f'{provider}-{stage}.log'
   if path.is_file():recorder.artifact(path,f'preflight/{stage}.log')
  for suffix in ('seed-profile.json','text-sanity.json'):
   path=root/f'{provider}-{suffix}'
   if path.is_file() and ('seed-search' if suffix.startswith('seed') else 'text-sanity') in executed:recorder.artifact(path,f'preflight/{suffix}')
  for chunk in (32,256,512):
   path=root/f'{provider}-seed-{chunk}'/'forward.json'
   if path.is_file():recorder.artifact(path,f'preflight/search-{chunk}.json')
  recorder.artifact(__file__,'preflight/runner.py')
  recorder.artifact(root/'seed-search.py','preflight/seed-search.py')
  recorder.artifact(root/'qualify-adaptive.py','preflight/qualify-adaptive.py')
  recorder.update(report,final=True,status='passed' if passed else 'failed')
 print(provider,'retained',recorder.id,flush=True)
subprocess.run([python,str(root/'ready.py')],check=True)

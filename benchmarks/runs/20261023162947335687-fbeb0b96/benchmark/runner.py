"""Matched short real-text acceptance audit against pinned llama.cpp DSpark."""
import hashlib,json,os,resource,signal,socket,statistics,subprocess,time,traceback,urllib.request
from pathlib import Path
from akas.benchmarking.history import Recorder,write,digest
ROOT=Path('/home/sagemaker-user/akas');OUT=ROOT/'build/lfm-moe/c4-final/llama';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'build/baselines/llama.cpp';BINARY=SOURCE/'build-cuda/bin/llama-server'
TARGET=ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
WORKLOAD=ROOT/'build/lfm-moe/c4-final/workload.json';workload=json.loads(WORKLOAD.read_text())
reference={}
for rid in ('20261023151432143989-67b63b5e','20261023151533293524-174061e4'):
 for row in json.loads((ROOT/'benchmarks/runs'/rid/'report.json').read_text())['results']:reference[row['text_case']['id']]=row
revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
env=dict(os.environ);env['LD_LIBRARY_PATH']=str(ROOT/'build/baselines/cuda12-libs/nvidia/cublas/lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')

target_digest=digest(TARGET);draft_digest=digest(TARGET.with_name('dspark-q8_0.gguf'));binary_digest=digest(BINARY)

for mode in ('ar','dspark'):
 dtype='q8_0';threshold=.5
 draft=TARGET.with_name('dspark-f16.gguf' if dtype=='float16' else 'dspark-q8_0.gguf')
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 command=[str(BINARY),'-m',str(TARGET),'-md',str(draft),'--spec-type','draft-dspark',
  '--spec-draft-n-max','9','--spec-draft-n-min','0','--spec-draft-p-min',str(threshold),
  '-ngl','99','-ngld','99','--device','CUDA0','-fa','on','-c','131072','-np','1','-b','2048','-ub','512',
  '-t','4','-tb','4','-ctk','f16','-ctv','f16','-ctkd','f16','-ctvd','f16',
  '--cache-ram','0','--ctx-checkpoints','0','--no-cache-idle-slots','--no-warmup','--metrics',
  '--host','127.0.0.1','--port',str(port),'--no-webui']
 if mode=='ar':
  for flag in ('-md','--spec-type','--spec-draft-n-max','--spec-draft-n-min','--spec-draft-p-min','-ngld','-ctkd','-ctvd'):
   i=command.index(flag);del command[i:i+2]
 meta=dict(runtime='llama',runtime_revision=revision,purpose='benchmark',protocol='llama-server-eval-real-text-v1',
  provider='cuda',target='sm_89',device=dict(name='NVIDIA L4'),model='LiquidAI/LFM2.5-24B-A2B',mode=mode,
  weight_format='q4_0',draft_weight_dtype=dtype,kv_dtype='float16',draft_kv_dtype='float16',confidence_threshold=threshold,
  block_size=9,window=10,context_capacity=131072,output_tokens=128,ignore_eos=True,warmup=0,repeats=1,
  checkpoint_sha256=target_digest,draft_checkpoint_sha256=draft_digest,binary_sha256=binary_digest,
  text_workload_sha256=digest(WORKLOAD),command=command,shape_timeout_seconds=180,
  qualification='Quick concatenated C4 continuation: same eight token-ID prefixes and Q4 target as AKAS, F16 target/draft KV, same Q8 draft in speculative mode, greedy with penalties disabled, 128 outputs ignoring EOS. One short untimed initialization request, no per-shape warmup and one measured sample. Phase rates use upstream server timing, not equivalent kernel timing. Raw accepted/proposed counters retained. No task accuracy claim.')
 if mode=='ar':meta.update(draft_weight_dtype=None,draft_checkpoint_sha256=None,draft_kv_dtype=None,confidence_threshold=None,block_size=None,window=None)
 rec=Recorder(ROOT/'benchmarks',meta,source_root=SOURCE,kind='configuration',parent='20261023151533293524-174061e4',change='Audit DSpark acceptance against official llama.cpp implementation on matched short prompts')
 report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);directory=OUT/mode;directory.mkdir(exist_ok=True);write(directory/'run-id.json',dict(run_id=rec.id));log_path=directory/'server.log';process=None;passed=False
 print('RUN',mode,dtype,threshold,rec.id,flush=True)
 def request(path,data=None,raw=False,timeout=180):
  req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=timeout) as response:return response.read().decode() if raw else json.load(response)
 def counters():
  lines=request('/metrics',raw=True).splitlines();result={}
  for line in lines:
   if line.startswith('#'):continue
   fields=line.split()
   if len(fields)==2 and 'spec_decode_num_' in fields[0]:result[fields[0].split(':')[-1]]=float(fields[1])
  return result
 try:
  with log_path.open('w') as output:
   process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env)
   until=time.monotonic()+600
   while True:
    if process.poll() is not None:raise RuntimeError(f'llama-server exited {process.returncode}; see server.log')
    try:
     if request('/health',timeout=2).get('status')=='ok':break
    except Exception:
     if time.monotonic()>until:raise TimeoutError('llama-server startup exceeded 600s')
     time.sleep(.25)
   request('/completion',dict(prompt=workload['cases'][0]['token_ids'],n_predict=4,temperature=0,top_k=1,repeat_penalty=1.,ignore_eos=True,cache_prompt=False))
   for item in workload['cases']:
    print('MEASURE',mode,dtype,threshold,item['id'],flush=True)
    payload=dict(prompt=item['token_ids'],n_predict=128,temperature=0,top_k=1,seed=17,
      repeat_penalty=1.,presence_penalty=0.,frequency_penalty=0.,ignore_eos=True,cache_prompt=False,return_tokens=True)
    rows=[];outputs=[]
    for iteration in range(1):
     before=counters();start=time.perf_counter();response=request('/completion',payload);wall=time.perf_counter()-start;after=counters()
     actual=response['tokens'];timings=response['timings']
     if len(actual)!=128 or response.get('tokens_predicted')!=128:raise ValueError('Incomplete llama.cpp output')
     if timings['prompt_n']!=len(item['token_ids']):raise ValueError('llama.cpp evaluated different prompt length')
     proposed=int(timings.get('draft_n',0));accepted=int(timings.get('draft_n_accepted',0))
     if (mode=='dspark' and proposed<1) or not 0<=accepted<=proposed:raise ValueError('Invalid speculative acceptance accounting')
     delta={k:after[k]-before.get(k,0.) for k in after}
     sample=dict(wall_seconds=wall,timings=timings,proposed=proposed,accepted=accepted,phase_protocol='llama-server-eval-real-text-v1',
      acceptance=accepted/proposed if proposed else 0.,counters=delta,output_ids=actual,
      prefill_tokens_per_second=len(item['token_ids'])/(timings['prompt_ms']/1000),
      decode_tokens_per_second=127/(timings['predicted_ms']/1000))
     rows.append(sample);outputs.append(actual)
    out=outputs[-1]
    row=dict(case=dict(batch=1,prompt=len(item['token_ids']),generated=128,depth=0,mixed=False),status='ok',
      text_case={k:v for k,v in item.items() if k!='token_ids'},prompt_sha256=item['prompt_sha256'],warmup=0,repeats=1,
      samples=rows,output_ids=[out],output_text=response['content'],deterministic=all(x==outputs[0] for x in outputs),
      acceptance=statistics.mean(x['acceptance'] for x in rows),proposed=rows[-1]['proposed'],accepted=rows[-1]['accepted'],
      akas_acceptance=None,akas_output_exact=None,
      akas_common_prefix=None)
    row['summary']={k:dict(median=statistics.median(x[k] for x in rows)) for k in ('prefill_tokens_per_second','decode_tokens_per_second')}
    report['results'].append(row);rec.update(report);write(directory/'report.json',report)
    print('RESULT',mode,dtype,threshold,item['id'],'ACCEPT',round(row['acceptance']*100,2),'COUNTS',row['accepted'],row['proposed'],'TPS',round(row['summary']['decode_tokens_per_second']['median'],2),'PREFIX_AKAS',row['akas_common_prefix'],'STEPS',rows[-1]['counters'],flush=True)
   passed=True
 except BaseException:
  meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
 finally:
  if process is not None and process.poll() is None:
   process.terminate()
   try:process.wait(timeout=15)
   except subprocess.TimeoutExpired:process.kill();process.wait()
  meta['cpu_peak_rss_bytes']=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)*1024
  rec.artifact(__file__,'benchmark/runner.py');rec.artifact(WORKLOAD,'benchmark/text-workload.json');rec.artifact(log_path,'baseline/server.log')
  for file in ('src/models/dflash.cpp','src/models/lfm2.cpp','common/speculative.cpp','tools/server/server-context.cpp'):rec.artifact(SOURCE/file,'baseline/source/'+file)
  write(directory/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
 if not passed:raise SystemExit(1)

"""Verify the measured binaries with existing operation recipes and model checks."""
from pathlib import Path
import json,sys,time
import modal
if Path('/workspace/akas').is_dir():sys.path.insert(0,'/workspace/akas')
else:sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.h200_modal import ROOT,cimage,volume,MODEL
app=modal.App('akas-h200-verify-existing')
RUN='20261102230421074409-17a02c7a'
PROGRAM='builtin-measured-225932'
SHA='17a04c41579da4c2c24224a22af9774fc39ae3817f8a9cd3beb6414fc504e9ab'

@app.function(image=cimage,gpu='H200',volumes={'/data':volume},timeout=900,cpu=8,memory=65536,scaledown_window=2,max_containers=1)
def verify():
 import gc,hashlib,os,tempfile,traceback,zipfile
 import numpy as np
 import tensor
 from tensor.compiler.search import ScheduleProfile
 from akas.artifacts.read import read_program,artifact_path
 from akas.graph import Graph
 from akas.compilation.compile import plan
 from akas.compilation.validate import kernel_bindings
 from akas.kernels.registry import validation,select
 from akas.kernels.contracts import mapped_validation
 from akas.benchmarking.history import Recorder,write,digest
 from akas.models.lfm2 import LFM2MoE
 from akas.optimization.evaluation import evaluate
 from akas.benchmarking.session import Session
 from akas.cli import checkpoint_digest
 os.chdir('/workspace/akas')
 os.environ['AKAS_SKIP_AUTOTUNE']='0'
 os.environ['AKAS_BENCHMARK_WARMUP']='1';os.environ['AKAS_BENCHMARK_REPEATS']='3'
 program=Path('/data/programs')/PROGRAM
 if digest(program/'program.json')!=SHA:raise ValueError('measured program changed')
 manifest=read_program(program)
 measured=json.loads((Path('/data/standard-benchmarks/runs')/RUN/'run.json').read_text())
 if measured['metadata']['program_sha256']!=SHA:raise ValueError('measurement program identity mismatch')
 if checkpoint_digest('/data/checkpoint')!=measured['metadata']['checkpoint_sha256']:raise ValueError('measured checkpoint changed')
 prior_id='20261103002207164569-1f09a073'
 prior_root=Path('/data/standard-benchmarks/runs')/prior_id
 prior_record=json.loads((prior_root/'run.json').read_text())
 prior_raw=prior_root/'operation-checks.json'
 expected_hash=next(a['sha256'] for a in prior_record['artifacts'] if a['path']=='operation-checks.json')
 if digest(prior_raw)!=expected_hash:raise ValueError('prior checks changed')
 previous_checks=json.loads(prior_raw.read_text())
 if previous_checks['program_sha256']!=SHA:raise ValueError('prior program differs')
 prior={(p['phase'],p['rows'],p['chunk'],k['key'],k['sha256']):k for p in previous_checks['phases'] for k in p['kernels'] if k.get('status')=='passed'}
 metadata=dict(runtime='akas',purpose='existing-program-verification',model=MODEL,provider='cuda',target=manifest['target'],
  weight_format='fp8',kv_dtype='float16',mode='ar',protocol='verification-only-v1',
  verified_run_id=RUN,program=PROGRAM,program_sha256=SHA,checkpoint_sha256=measured['metadata']['checkpoint_sha256'],
  tuning=False,compilation=False,scope='Only P2048/G128/C1,4,16,64,256 with chunk 2048; intermediate-prefill and unused compiled geometries are excluded',prior_operation_checks_run_id=prior_id,acceptance='Unchanged operation-owned references and comparators; existing model evaluator checks complete measured greedy outputs, slot reuse and finite diagnostics; retained independent BOS/chat first-token checks.')
 rec=Recorder('/data/standard-benchmarks',metadata,parent=RUN,kind='qualification',
  change='Verify the exact last-measured H200 program; retain all original timing samples',
  dependency_sources={'tensor':Path('/workspace/tensor')})
 rec.artifact(Path(__file__),'verify.py')
 operations=[];phases=[];model_results=[];independent=[];diagnostics={};observed=set();seen=set();status='failed';tick=time.monotonic()
 print('VERIFY EXISTING',rec.id,RUN,SHA,flush=True)
 try:
  with tensor.Device() as device:
   metadata['device']=dict(device.info)
   used=[p for p in manifest['phases'] if p['phase']=='decode' and p['rows'] in (1,4,16,64,256) or p['phase']=='prefill' and p['rows'] in (1,4,16)]
   if manifest.get('reset'):used.append(dict(phase='reset',rows=1,chunk=1,**manifest['reset']))
   for phase in used:
    forward=json.loads((program/phase['path']).read_text());graph=Graph.from_metadata(forward['forward'])
    profile=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target=manifest['target'],entries=[
     dict(operation=k['kind'],parameters=k['parameters'],schedule=k.get('schedule') or {}) for k in forward['kernels'].values()]))
    bundle=plan(graph,provider='cuda',target=manifest['target'],schedules=profile)
    mapping={k.key:k.mapping for k in bundle.kernels}
    entry=dict(phase=phase['phase'],rows=phase['rows'],chunk=phase['chunk'],kernels=[])
    phases.append(entry)
    for key,k in forward['kernels'].items():
     path=artifact_path(program,k)
     task=dict(key=key,phase=phase['phase'],rows=phase['rows'],chunk=phase['chunk'],kind=k['kind'],sha256=k['sha256'])
     entry['kernels'].append(task)
     signature=json.dumps([k['sha256'],k['kind'],k['parameters'],k.get('schedule'),mapping.get(key)],sort_keys=True)
     with device.load(path) as module:
      kernel_bindings(graph,key,module.manifest['arguments'])
      if signature in seen:task.update(status='passed',reused_identical_artifact=True);continue
      cached=prior.get((phase['phase'],phase['rows'],phase['chunk'],key,k['sha256']))
      if cached is not None:
       task.update(cached,reused_prior_run_id=prior_id);seen.add(signature)
       continue
      print('CHECK',phase['phase'],phase['rows'],k['kind'],key,flush=True)
      schedule=dict(k.get('schedule') or {});preferred=schedule.pop('implementation',None)
      impl=select(k['kind'],k['parameters'],preferred=preferred,contract=graph.contract,provider='cuda',phase=graph.phase,target=manifest['target'])
      parameters={**dict(impl.defaults),**k['parameters'],**schedule}
      args=module.manifest['arguments'];adapter=mapping.get(key)
      case=validation(k['kind'],parameters,args) if adapter is None else mapped_validation(k['kind'],parameters,args,adapter)
      buffers={}
      try:
       dtypes={a['name']:a['dtype'] for a in args}
       for name,value in case.arguments.items():buffers[name]=device.from_numpy(value,dtype='bfloat16') if dtypes[name]=='bfloat16' else device.from_numpy(value)
       module.launch(*(buffers[a['name']] for a in args));device.synchronize()
       checked=[]
       for name,expected in case.expected.items():
        written=case.arguments[name] is not expected and not np.array_equal(case.arguments[name],expected)
        if not written and buffers[name].nbytes>16*1024*1024:continue
        actual=buffers[name].to_numpy()
        if name in case.comparators:case.comparators[name](actual,expected)
        elif actual.dtype.kind in 'biu' or not written:np.testing.assert_array_equal(actual,expected)
        else:
         a=actual.reshape(-1);e=expected.reshape(-1)
         for start in range(0,a.size,4194304):np.testing.assert_allclose(a[start:start+4194304],e[start:start+4194304],rtol=case.rtol,atol=case.atol,equal_nan=False)
        checked.append(name)
       task.update(status='passed',checked=checked,rtol=case.rtol,atol=case.atol)
       seen.add(signature)
      finally:
       for b in reversed(tuple(buffers.values())):b.release()
       del case,buffers
     operations.append(task)
     write(rec.directory/'operation-checks.json',dict(program_sha256=SHA,phases=phases))
    print('PHASE PASSED',phase['phase'],phase['rows'],len(entry['kernels']),flush=True)
    gc.collect()
   metadata['operation_instances']=sum(len(p['kernels']) for p in phases);metadata['unique_artifacts_checked']=len(seen)
   with LFM2MoE('/data/checkpoint',program,device,graphs=True,checkpoint_options={'tokenizer':False}) as model:
    model.prefill_group_size=16
    original_select=model.batches.select
    def selected(count,chunk,**kwargs):
     frame=original_select(count,chunk,**kwargs)
     if frame.phase=='prefill_state':raise ValueError('excluded intermediate-prefill phase executed')
     observed.add((frame.phase,frame.rows,frame.chunk))
     return frame
    model.batches.select=selected
    with tempfile.TemporaryDirectory() as temporary:
     refs=Path(temporary)
     with zipfile.ZipFile('/data/qualification/lfm2.5-24b-references.zip') as z:z.extractall(refs)
     provenance=json.loads((refs/'provenance.json').read_text())
     for item in provenance['source_files']:
      if digest(refs/item.get('retained_as',item['file']))!=item['sha256']:raise ValueError('reference checksum mismatch')
     references={name:json.loads((refs/(name+'_results.json')).read_text()) for name in ('akas','vllm','sglang')}
     for suite in ('bos','chat'):
      for index,prompt in enumerate(references['akas'][suite]):
       expected=[references[name][suite][index] for name in ('vllm','sglang')]
       if any(p['tokens']!=prompt['tokens'] for p in expected):raise ValueError('reference prompts differ')
       with Session(model,batch=1,chunk=2048) as session:
        session.reset();first=int(session.prefill([prompt['tokens']])[0]);finite=bool(np.isfinite(session.read_feature('logits')).all())
       passed=finite and all(first==p['first_token'] for p in expected)
       independent.append(dict(suite=suite,index=index,first_token=first,reference_tokens=[p['first_token'] for p in expected],passed=passed))
       print('INDEPENDENT',suite,index,passed,flush=True)
       if not passed:raise ValueError('retained independent first-token check failed')
     metadata['independent_reference_run_id']=provenance['qualification_run_id']
    recipe=dict(chunk=2048,window=None,graphs=True,prefill_requests=256,prefill_token_budget=524288,decode_prefill_token_budget=256,mode='ar')
    measured['metadata']['run_id']=RUN
    print('CHECK ALL FIVE MEASURED OUTPUTS',flush=True)
    model_results,diagnostics,statistics=evaluate(model,recipe,dict(batches=[1,4,16,64,256],prefix_tokens=[2048],output_tokens=128),reference=measured)
    if len(model_results)!=5 or any(r['status']!='ok' or not r['correctness']['correct'] for r in model_results):raise ValueError('model workload checks failed')
    metadata['model_checks']=statistics
   if digest(program/'program.json')!=SHA:raise ValueError('verification changed measured program')
   status='passed'
 except Exception as e:
  metadata.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());print(metadata['traceback'],flush=True)
  if getattr(e,'results',None):model_results=e.results
 finally:
  metadata.update(qualification_status=status,elapsed_seconds=time.monotonic()-tick)
  write(rec.directory/'operation-checks.json',dict(program_sha256=SHA,phases=phases))
  write(rec.directory/'independent-checks.json',dict(cases=independent))
  write(rec.directory/'executed-phases.json',dict(phases=sorted(observed),excluded_phase='prefill_state',excluded_phase_calls=0))
  np.savez_compressed(rec.directory/'diagnostics.npz',**diagnostics)
  for name in ('operation-checks.json','independent-checks.json','diagnostics.npz','executed-phases.json'):rec.artifact(rec.directory/name,name)
  record=rec.update(dict(metadata=metadata,results=model_results),final=True,status=status)
  volume.commit()
 print('VERIFIED',rec.id,status,metadata.get('operation_instances'),flush=True)
 return dict(run_id=rec.id,status=status,error=metadata.get('error'),verified_run_id=RUN,program_sha256=SHA)

@app.local_entrypoint()
def main():
 ledger_path=ROOT/'build/h200/cost-ledger.json';ledger=json.loads(ledger_path.read_text());rate=.001261+8*.0000131+64*.00000222
 if sum(r['conservative_usd'] for r in ledger)+930*rate>30:raise RuntimeError('credit limit')
 row=dict(action='verify-existing-program',status='reserved',started_at=time.time(),conservative_usd=930*rate);ledger.append(row);ledger_path.write_text(json.dumps(ledger,indent=2));tick=time.monotonic()
 try:
  result=verify.remote();row.update(status='completed',result=result);print(json.dumps(result),flush=True)
 finally:
  row['elapsed_seconds']=time.monotonic()-tick
  if row['status']=='completed':row['conservative_usd']=(row['elapsed_seconds']+30)*rate
  ledger_path.write_text(json.dumps(ledger,indent=2))

"""Additional explicit experiments for the persistent verification lab."""
from contextlib import ExitStack
import json
from pathlib import Path
import numpy as np
from tensor.compiler.tuning import measure_cuda
from akas.execution.plan import ExecutionPlan,bind_named,call_bindings
from akas.benchmarking.cases import Case
from akas.benchmarking.runner import run
from akas.benchmarking.speculative import SpeculativeSession

def execute(command,lab,rec,report):
 device,target,head,decoder=(lab[n] for n in ('device','target','head','decoder'))
 meta=report['metadata'];directory=Path(command['directory']);compiled=json.loads((directory/'compilation.json').read_text())
 meta['command']=command
 if command['action']=='measure':
  original_graphs={};variant_graphs={}
  with ExitStack() as owned:
   selected=[r for r in compiled if r['name'].split('-r')[1] in command['geometries']]
   modules={}
   for row in selected:
    module=device.load(directory/(row['name']+'.tbin'));owned.callback(module.release)
    modules[(row['rows'],row['kind'])]=module;rec.artifact(directory/(row['name']+'.tbin'),'kernels/'+row['name']+'.tbin')
   for chunk in sorted({r['rows'] for r in selected}):
    frame=target.batches.select(1,chunk,phase='verify');f=frame.forward;original_graphs[chunk]=f.graph
    plan=ExecutionPlan(device,labeled=True)
    for call,entry in zip(f.definition.calls,f.plan):
     if call.kind=='quantized_experts':
      module=modules[(chunk,'down' if call.parameters['routed_input'] else 'gate')]
      a,b=bind_named(device,module,call_bindings(entry[0],entry[1]));plan.append((a,b,entry[2]))
     else:plan.append(entry)
    graph=plan.capture(resources=tuple(f.buffers.values())+tuple(f.kernels.values())+tuple(modules.values()));variant_graphs[chunk]=graph;owned.callback(graph.close)
   baseline={}
   try:
    for variant in ('baseline','candidate'):
     for chunk,graph in (original_graphs if variant=='baseline' else variant_graphs).items():target.batches.select(1,chunk,phase='verify').forward.graph=graph
     for item in lab['workload']['cases']:
      if command.get('cases') and item['id'] not in command['cases']:continue
      print('DECODE',variant,item['id'],flush=True)
      with SpeculativeSession(decoder,batch=1,chunk=512,timeout=120) as session:
       row=run(session,Case(1,len(item['token_ids']),128),prompts=[item['token_ids']],warmup=1,repeats=3)
       output=list(session.requests[0].output);stats=session.stats
      row.update(variant=variant,text_case={k:v for k,v in item.items() if k!='token_ids'},output_ids=[output],speculation=stats)
      if variant=='baseline':baseline[item['id']]=row
      else:
       old=baseline[item['id']]
       if output!=old['output_ids'][0]:raise ValueError('Candidate changes speculative output')
       row['baseline_decode_tps']=old['summary']['decode_tokens_per_second']['median'];row['improvement_over_baseline']=row['summary']['decode_tokens_per_second']['median']/row['baseline_decode_tps'];row['baseline_output_exact']=True
      report['results'].append(row);rec.update(report)
      print('DECODE_RESULT',variant,item['id'],row['summary']['decode_tokens_per_second']['median'],stats,flush=True)
   finally:
    for chunk,graph in original_graphs.items():target.batches.select(1,chunk,phase='verify').forward.graph=graph
   meta['qualification']='Complete experimental DSpark decoding, 128 outputs on fixed real-text short prompts, Q8 draft and F16 KV. One warmup, three samples. Only qualified expert kernel overlays are changed; original programs retained. All candidate outputs must equal the unchanged speculative baseline. Public compilation and serving promotion remains a separate gate.'
 elif command['action']=='fused-micro':
  shots=lab['snapshots']
  if not shots:raise ValueError('Capture snapshots first')
  frame=target.batches.select(1,10,phase='verify')
  activation=next(entry[0] for call,entry in zip(frame.forward.definition.calls,frame.forward.plan) if call.kind=='swiglu_experts')
  checks=[]
  candidates=[('baseline-before',None),*((r['name'],r) for r in compiled if not command.get('geometries') or r['name'] in command['geometries']),('baseline-after',None)]
  for name,row in candidates:
   with ExitStack() as owned:
    module=None
    if row:
     module=device.load(directory/(row['name']+'.tbin'));owned.callback(module.release)
    plan=ExecutionPlan(device);resources=[]
    for start in range(0,len(shots),3):
     gate,up,down=shots[start:start+3]
     if (gate['parameters']['routed_input'] or up['parameters']['routed_input'] or not down['parameters']['routed_input']):raise ValueError('Expected private gate/up/activation/down sequence')
     if module:
      values=dict(gate['values'],w2=up['values']['w'],out=down['values']['x']);plan.append(bind_named(device,module,values));resources.extend((module,*values.values()))
     else:
      for shot in (gate,up):plan.append(bind_named(device,shot['kernel'],shot['values']))
      plan.bind(activation,(gate['values']['out'],up['values']['out'],down['values']['x']));resources.append(activation)
     plan.append(bind_named(device,down['kernel'],down['values']))
     for shot in (gate,up,down):resources.extend((shot['kernel'],*shot['values'].values()))
    with plan.capture(resources=tuple(dict.fromkeys(resources))) as graph:
     graph.launch();device.synchronize();bitwise=True
     for down in shots[2::3]:
      actual=down['values']['out'].to_numpy();np.testing.assert_allclose(actual,down['expected'],rtol=.002,atol=.0005);bitwise=bitwise and np.array_equal(actual.view('uint32'),down['expected'].view('uint32'))
     timing=measure_cuda(device,graph.launch,warmup=2,samples=7,repeats=2)
    check=dict(geometry=name,timing=timing,down_outputs_correct=True,down_outputs_bitwise=bool(bitwise));checks.append(check);meta['checks']=checks;rec.update(report)
    print('FUSED_MICRO',name,timing['median_gpu_seconds']*1000,'BITWISE',bitwise,flush=True)
    if row:rec.artifact(directory/(row['name']+'.tbin'),'kernels/'+row['name']+'.tbin')
  meta.update(snapshot_run=lab['snapshot_run'],qualification='Full streamed expert gate/up/SwiGLU/down sequences on live routed snapshots. Fused candidate binds its activation directly into the unchanged down projection. All final expert outputs checked against the original verifier. Kernel diagnostic, not decode throughput.')
 elif command['action']=='planar-micro':
  shots=lab['snapshots']
  if not shots:raise ValueError('Capture snapshots first')
  weights=list({s['values']['w'].pointer:s['values']['w'] for s in shots}.values())
  count=weights[0].nbytes
  if any(w.nbytes!=count for w in weights):raise ValueError('Transfer artifact requires uniform bank sizes')
  with ExitStack() as owned:
   transfer_root=lab['ROOT']/'build/lfm-moe/planar-q4'
   pack=device.load(transfer_root/'pack.tbin');unpack=device.load(transfer_root/'unpack.tbin');owned.callback(pack.release);owned.callback(unpack.release)
   temporary=device.empty((count,),'uint8');owned.callback(temporary.release)
   converted=[]
   def transform(module,bank):module.launch(bank,temporary);bank.copy_from(temporary)
   baseline=ExecutionPlan(device)
   resources=[]
   for s in shots:baseline.append(bind_named(device,s['kernel'],s['values']));resources.extend((s['kernel'],*s['values'].values()))
   checks=[]
   with baseline.capture(resources=tuple(dict.fromkeys(resources))) as original:
    original_timing=measure_cuda(device,original.launch,warmup=2,samples=7,repeats=2)
    checks.append(dict(geometry='raw-before',timing=original_timing));print('PLANAR_MICRO','raw-before',original_timing['median_gpu_seconds']*1000,flush=True)
    try:
     for bank in weights:transform(pack,bank);converted.append(bank)
     device.synchronize()
     for geometry in sorted({r['name'].split('-r')[1] for r in compiled}):
      if command.get('geometries') and geometry not in command['geometries']:continue
      selected=[r for r in compiled if r['name'].split('-r')[1]==geometry]
      with ExitStack() as modules_owned:
       modules={}
       for r in selected:
        module=device.load(directory/(r['name']+'.tbin'));modules_owned.callback(module.release);modules[r['kind']]=module
       plan=ExecutionPlan(device);resources=[]
       for s in shots:
        module=modules['down' if s['parameters']['routed_input'] else 'gate'];plan.append(bind_named(device,module,s['values']));resources.extend((module,*s['values'].values()))
       with plan.capture(resources=tuple(dict.fromkeys(resources))) as graph:
        graph.launch();device.synchronize();bitwise=True
        for s in shots:
         actual=s['values']['out'].to_numpy();np.testing.assert_allclose(actual,s['expected'],rtol=.002,atol=.0005);bitwise=bitwise and np.array_equal(actual.view('uint32'),s['expected'].view('uint32'))
        timing=measure_cuda(device,graph.launch,warmup=2,samples=7,repeats=2)
       checks.append(dict(geometry=geometry,timing=timing,bitwise=bool(bitwise),full_outputs_correct=True));meta['checks']=checks;rec.update(report)
       print('PLANAR_MICRO',geometry,timing['median_gpu_seconds']*1000,'BITWISE',bitwise,flush=True)
       for r in selected:rec.artifact(directory/(r['name']+'.tbin'),'kernels/'+r['name']+'.tbin')
    finally:
     for bank in converted:transform(unpack,bank)
     device.synchronize()
    original.launch();device.synchronize()
    for s in shots:np.testing.assert_array_equal(s['values']['out'].to_numpy().view('uint32'),s['expected'].view('uint32'))
    after=measure_cuda(device,original.launch,warmup=2,samples=7,repeats=2);checks.append(dict(geometry='raw-after',timing=after));meta['checks']=checks
    print('PLANAR_MICRO','raw-after',after['median_gpu_seconds']*1000,flush=True)
   for name in ('pack','unpack'):rec.artifact(transfer_root/(name+'.tbin'),'layout/'+name+'.tbin')
  meta.update(snapshot_run=lab['snapshot_run'],weight_tensors=len(weights),payload_bytes=sum(w.nbytes for w in weights),
              qualification='Whole streamed expert-bank replay with lossless Q4 payload/scale separation. Raw baseline before/after in-place device conversion; conversion is excluded from timings and reversed before returning to model execution. All full outputs checked, including bitwise raw restoration. Actual deployment would prepare this immutable layout once at upload.')
 else:raise ValueError('Unknown action')
 rec.artifact(directory/'compilation.json','kernels/compilation.json')

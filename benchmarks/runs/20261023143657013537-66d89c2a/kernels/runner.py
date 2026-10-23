"""Qualify paired gate/up/SwiGLU at short routed verification shapes."""
from contextlib import ExitStack
import json
from pathlib import Path
import resource
import traceback
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.artifacts.schema import Source
from akas.benchmarking.history import Recorder,write
from akas.checkpoints.gguf import GGUF
from akas.compilation.emit import build_artifact
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.memory.transfer import write as transfer
from akas.ops.moe.validation import tile_tables
from akas.quantization.weights.ggml.validation import decode_rows

ROOT=Path('/home/sagemaker-user/akas');OUT=ROOT/'build/lfm-moe/dspark-fused-short-geometry';OUT.mkdir(exist_ok=True)
PROGRAM=ROOT/'build/lfm-moe/dspark-short-optimized/target';CP=ROOT/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
top=read_program(PROGRAM);manifest=json.loads((PROGRAM/next(p['path'] for p in top['phases'] if p['phase']=='verify' and p['chunk']==10)).read_text())
record=next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==2048)
activation=next(k for k in manifest['kernels'].values() if k['kind']=='swiglu_experts')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4'),
 weight_format='q4_0',mode='dspark',kv_dtype='float16',model='akas.models.lfm2.model:LFM2MoE',checks=[],
 qualification='Actual packed Q4 gate/up banks with balanced, skewed and fully active ten-token routing. Independent sampled F64 products with F16 operands; full-output reference is the qualified three-kernel gate/up/SwiGLU sequence. Inactive entries retained. Kernel-only diagnostic; full-model verification required.')
rec=Recorder(ROOT/'benchmarks',meta,kind='algorithm',parent='20261023141118005047-055399b4',change='Qualify multi-token fused expert gate/up/SwiGLU with codec-owned dequantization')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False;compiled=[];write(OUT/'run-id.json',dict(run_id=rec.id));print('RUN',rec.id,flush=True)
try:
 for bm,bn,bk,threads in ((16,32,128,128),(16,64,128,128),(16,64,64,128),(16,128,64,128),(16,64,128,256)):
  p={**record['parameters'],**dict(block_m=bm,columns=bn,block_k=bk,threads=threads,output_dtype='float32')}
  name=f'fused-r10-{bm}-{bn}-{bk}-{threads}'
  source=OUT/(name+'.py');source.write_text(Source('fused_expert_activation','factory',(p,)).render());build_artifact(source,source.with_suffix('.tbin'),provider='cuda',target='sm_89')
  compiled.append(dict(name=name,parameters=p));write(OUT/'compilation.json',compiled);print('BUILT',name,flush=True)
 gguf=GGUF(CP)
 with tensor.Device(provider='cuda') as dev,ExitStack() as owned:
  projection=dev.load(artifact_path(PROGRAM,record));owned.callback(projection.release)
  swiglu=dev.load(artifact_path(PROGRAM,activation));owned.callback(swiglu.release)
  arrays=sample_arguments(projection.manifest['arguments']);arrays['x']*=8
  name=sorted(n for n in gguf.tensors if n.endswith('ffn_gate_exps.weight'))[19]
  arrays['w'][:]=gguf.packed(name);gguf.discard(name);arrays['w2']=gguf.packed(name.replace('ffn_gate_exps','ffn_up_exps')).copy();gguf.discard(name.replace('ffn_gate_exps','ffn_up_exps'))
  arrays['out'].fill(1.25);buffers={n:dev.from_numpy(a) for n,a in arrays.items()}
  for b in buffers.values():owned.callback(b.release)
  gate=dev.empty(arrays['out'].shape,'float32');up=dev.empty(arrays['out'].shape,'float32');owned.callback(gate.release);owned.callback(up.release)
  variants=[]
  for row in compiled:
   module=dev.load(OUT/(row['name']+'.tbin'));owned.callback(module.release);variants.append((row['name'],module))
  for layout in ('balanced','skewed','full'):
   ids=(np.arange(10)[:,None]*17+np.arange(4)[None,:]*13)%64 if layout!='skewed' else np.column_stack((np.zeros(10,dtype='int32'),np.ones(10,dtype='int32'),2+np.arange(10)%62,2+(np.arange(10)+17)%62))
   arrays['counts'].fill(0);arrays['routes'].fill(-1);independent=[]
   for expert in range(64):
    owners,ranks=np.where((ids if layout=='full' else ids[:-1])==expert);arrays['counts'][expert]=len(owners);arrays['routes'][expert,:len(owners)]=owners*4+ranks
    if not len(owners):continue
    owner,rank=int(owners[0]),int(ranks[0]);x=arrays['x'][owner].astype('float16').astype('float64');cols=np.unique(np.linspace(0,1535,33,dtype='int32'));reference=[]
    for col in cols:
     g=float(decode_rows(arrays['w'],2,2048,expert*1536+int(col),expert*1536+int(col)+1).astype('float16').astype('float64')[0]@x)
     u=float(decode_rows(arrays['w2'],2,2048,expert*1536+int(col),expert*1536+int(col)+1).astype('float16').astype('float64')[0]@x)
     reference.append(g/(1+np.exp(-g))*u)
    independent.append((owner,rank,cols,np.asarray(reference,dtype='float32')))
   arrays['tile_experts'][:],arrays['tile_rows'][:]=tile_tables(arrays['counts'],arrays['tile_experts'].shape[1])
   for field in ('counts','routes','tile_experts','tile_rows'):transfer(dev,buffers[field],arrays[field])
   baseline=ExecutionPlan(dev);baseline.bind(projection,tuple(gate if a['name']=='out' else buffers[a['name']] for a in projection.manifest['arguments']));baseline.bind(projection,tuple(up if a['name']=='out' else buffers['w2'] if a['name']=='w' else buffers[a['name']] for a in projection.manifest['arguments']));baseline.bind(swiglu,(gate,up,buffers['out']))
   with baseline.capture(resources=tuple(buffers.values())+(gate,up,projection,swiglu)) as original:
    transfer(dev,buffers['out'],arrays['out']);original.launch();expected=buffers['out'].to_numpy()
    for label,module in [('baseline-before',None),*variants,('baseline-after',None)]:
     transfer(dev,buffers['out'],arrays['out'])
     if module:
      plan=ExecutionPlan(dev);plan.bind(module,tuple(buffers[a['name']] for a in module.manifest['arguments']));graph=plan.capture(resources=tuple(buffers.values())+(module,))
     else:graph=original
     try:
      graph.launch();actual=buffers['out'].to_numpy();np.testing.assert_allclose(actual if layout=='full' else actual[:-1],expected if layout=='full' else expected[:-1],rtol=.002,atol=.0005)
      for owner,rank,cols,reference in independent:np.testing.assert_allclose(actual[owner,rank,cols],reference,rtol=.002,atol=.0002)
      if layout!='full' and module:np.testing.assert_array_equal(actual[-1],arrays['out'][-1])
      timing=measure(dev,graph.launch,warmup=3,samples=7,repeats=3)
     finally:
      if module:graph.close()
     check=dict(layout=layout,geometry=label,bitwise=bool(np.array_equal(actual.view('uint32'),expected.view('uint32'))),correct=True,timing=timing);meta['checks'].append(check);rec.update(report);print('PASS',layout,label,timing['median_seconds']*1000,'BITWISE',check['bitwise'],flush=True)
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 rec.artifact(__file__,'kernels/runner.py');rec.artifact(ROOT/'build/lfm-moe/fused_expert_activation.py','kernels/factory.py')
 if (OUT/'compilation.json').exists():rec.artifact(OUT/'compilation.json','kernels/compilation.json')
 for p in OUT.glob('*.tbin'):rec.artifact(p,'kernels/'+p.name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

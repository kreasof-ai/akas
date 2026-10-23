"""Bounded short-prefill geometry qualification using the public packed factory."""
from contextlib import ExitStack
import json
from pathlib import Path
import resource
import traceback
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.artifacts.read import artifact_path,read_program
from akas.artifacts.schema import Source
from akas.benchmarking.history import Recorder,write
from akas.checkpoints.gguf import GGUF
from akas.compilation.emit import build_artifact
from akas.compilation.selection import selected_source
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.kernels.definition import definition
from akas.memory.transfer import write as transfer
from akas.ops.moe.validation import tile_tables
from akas.quantization.weights.ggml.validation import decode_rows

ROOT=Path('/home/sagemaker-user/akas/build/lfm-moe/dspark-register-mma-geometry');ROOT.mkdir(exist_ok=True)
PROGRAM=Path('/home/sagemaker-user/akas/build/lfm-moe/dspark-short-optimized/target')
CHECKPOINT=Path('/home/sagemaker-user/akas/build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf')
index=read_program(PROGRAM)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',
 device=dict(name='NVIDIA L4',provider='cuda'),checkpoint=str(CHECKPOINT),
 checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
 operand_dtype='float16',accumulator_dtype='float32',shape_timeout_seconds=120,
 qualification='Prototype generic codec-backed register MMA projection on actual Q4_0 banks, with balanced, skewed and fully active routing. Independent sampled F64 reference, FP16 operand rounding, full-output tolerance against tensor-core baseline and inactive sentinel preservation. No model-specific constants or weight conversion. Accumulation order differs from the tensor-core reference. Kernel-only evidence, not throughput proof.')
parent=json.loads(Path('/home/sagemaker-user/akas/build/lfm-moe/cuda-f16-prepared-full/cuda/run-id.json').read_text())['run_id']
rec=Recorder('/home/sagemaker-user/akas/benchmarks',meta,kind='geometry',parent=parent,
 change='Test generic register-fed tensor-core expert projection for small verification blocks')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
passed=False;compiled=[]
try:
 for rows in (10,):
  choices=[(16,32,128,128,1),(16,32,128,128,2),(16,32,128,256,1),(16,32,128,256,2)]
  manifest=json.loads((PROGRAM/next(p['path'] for p in index['phases'] if p['phase']=='verify' and p['chunk']==rows)).read_text())
  for kind,depth,columns in (('gate',2048,1536),('down',1536,2048)):
   record=next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==depth)
   for storage_layout in ('raw',):
    parameters=json.loads(json.dumps(record['parameters']))
    if storage_layout=='planar':
     for part in parameters['weight_parts']:part['options']['layout']='planar'
    spec=dict(kind='quantized_experts',parameters=parameters,contract='reference',phase='verify')
    for bm,bn,bk,threads,stages in choices:
     axes=dict(threads=threads,unroll=stages)
     implementation,schedule=None,axes
     name=f'{storage_layout}-{kind}-r{rows}-{bm}-{bn}-{bk}-{threads}-s{stages}';source=ROOT/(name+'.py')
     source.write_text(Source('register_mma_experts','make',(spec,axes)).render())
     build_artifact(source,source.with_suffix('.tbin'),provider='cuda',target='sm_89')
     compiled.append(dict(name=name,kind=kind,rows=rows,parameters=parameters,schedule=axes,storage_layout=storage_layout))
     write(ROOT/'compilation.json',compiled);print('BUILT',name,flush=True)
 gguf=GGUF(CHECKPOINT)
 with tensor.Device(provider='cuda') as device:
  for rows in (10,):
   manifest=json.loads((PROGRAM/next(p['path'] for p in index['phases'] if p['phase']=='verify' and p['chunk']==rows)).read_text())
   for kind,depth,columns in (('gate',2048,1536),('down',1536,2048)):
    old=artifact_path(PROGRAM,next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==depth))
    names=sorted(n for n in gguf.tensors if n.endswith(f'ffn_{kind}_exps.weight'));weight_name=names[len(names)//2]
    with ExitStack() as owned:
     baseline=device.load(old);owned.callback(baseline.release)
     arrays=sample_arguments(baseline.manifest['arguments']);arrays['x']*=8
     arrays['w'][:]=gguf.packed(weight_name);gguf.discard(weight_name)
     arrays['out'].fill(1.25)
     buffers={n:device.from_numpy(a) for n,a in arrays.items()}
     for buffer in buffers.values():owned.callback(buffer.release)
     from akas.quantization.weights.ggml.kernel import repack_host
     planar_w=None
     variants=[]
     for item in compiled:
      if item['kind']!=kind or item['rows']!=rows:continue
      module=device.load(ROOT/(item['name']+'.tbin'));owned.callback(module.release);variants.append((item['name'],module,item['storage_layout']))
     for layout in ('balanced','skewed','full'):
      if layout in ('balanced','full'):ids=(np.arange(rows)[:,None]*17+np.arange(4)[None,:]*13)%64
      else:ids=np.column_stack((np.zeros(rows,dtype='int32'),np.ones(rows,dtype='int32'),2+np.arange(rows)%62,2+(np.arange(rows)+17)%62))
      arrays['counts'].fill(0);arrays['routes'].fill(-1);sampled=[]
      for expert in range(64):
       owners,ranks=np.where((ids if layout=='full' else ids[:-1])==expert)
       arrays['counts'][expert]=len(owners);arrays['routes'][expert,:len(owners)]=owners*4+ranks
       if not len(owners):continue
       owner,rank=int(owners[0]),int(ranks[0]);source=(arrays['x'][owner,rank] if kind=='down' else arrays['x'][owner]).astype('float16').astype('float64')
       selected=np.unique(np.linspace(0,columns-1,33,dtype='int32'));expected=np.empty(len(selected),dtype='float32')
       for i,col in enumerate(selected):
        w=decode_rows(arrays['w'],2,depth,expert*columns+int(col),expert*columns+int(col)+1).astype('float16').astype('float64')[0]
        expected[i]=w@source
       sampled.append((owner,rank,selected,expected))
      tables,tile_rows=tile_tables(arrays['counts'],arrays['tile_experts'].shape[1]);arrays['tile_experts'][:]=tables;arrays['tile_rows'][:]=tile_rows
      for field in ('counts','routes','tile_experts','tile_rows'):transfer(device,buffers[field],arrays[field])
      expected=None
      for geometry,module,storage_layout in [('original-before',baseline,'raw'),*variants,('original-after',baseline,'raw')]:
       transfer(device,buffers['out'],arrays['out'])
       supplied=dict(buffers,w=planar_w if storage_layout=='planar' else buffers['w'])
       args=tuple(supplied[a['name']] for a in module.manifest['arguments'])
       plan=ExecutionPlan(device);plan.bind(module,args);graph=plan.capture(resources=tuple(buffers.values())+(module,))
       try:
        graph.launch();actual=buffers['out'].to_numpy()
        for owner,rank,selected,independent in sampled:np.testing.assert_allclose(actual[owner,rank,selected],independent,rtol=.002,atol=.0002)
        if expected is None:expected=actual.copy()
        np.testing.assert_allclose(actual,expected,rtol=.002,atol=.0005)
        if layout!='full':np.testing.assert_array_equal(actual[-1],arrays['out'][-1])
        timing=measure(device,graph.launch,warmup=3,samples=7,repeats=3)
       finally:graph.close()
       check=dict(rows=rows,kind=kind,layout=layout,geometry=geometry,storage_layout=storage_layout,bitwise=bool(np.array_equal(actual.view('uint32'),expected.view('uint32'))),
        independent_reference_rows=len(sampled),independent_columns=33,inactive_unchanged=layout!='full',timing=timing)
       meta.setdefault('checks',[]).append(check);write(ROOT/'report.json',report)
       print('PASS',rows,kind,layout,geometry,timing['median_seconds']*1000,flush=True)
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 rec.artifact(__file__,'kernels/runner.py')
 rec.artifact(Path(__file__).with_name('register_mma_experts.py'),'kernels/factory.py')
 if (ROOT/'compilation.json').exists():rec.artifact(ROOT/'compilation.json','kernels/compilation.json')
 for path in ROOT.glob('*.tbin'):rec.artifact(path,'kernels/'+path.name)
 write(ROOT/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
 print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

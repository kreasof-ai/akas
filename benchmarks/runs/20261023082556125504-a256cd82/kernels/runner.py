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

ROOT=Path('build/lfm-moe/short-expert-tiles');ROOT.mkdir(exist_ok=True)
PROGRAM=Path('build/lfm-moe/cuda-f16-full/cuda/program')
CHECKPOINT=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf')
index=read_program(PROGRAM)
choices=[(16,32,64,128),(16,32,128,128),(16,64,64,128),(16,64,128,128),
 (16,128,64,128),(16,128,128,128),(16,128,128,256),(32,128,128,256),
 (64,128,128,256)]
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',
 device=dict(name='NVIDIA L4',provider='cuda'),checkpoint=str(CHECKPOINT),
 checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
 operand_dtype='float16',accumulator_dtype='float32',shape_timeout_seconds=120,
 qualification='Current public CUDA compact expert factory, actual Q4_0 midpoint banks, synthetic balanced and skewed routing at 32 and 512 prompt tokens. Sampled independent F64 products, established FP16 operand rounding, whole-output bitwise baseline agreement, inactive sentinel preservation. Kernel-only evidence; model and HTTP qualification follows selection.')
parent=json.loads(Path('build/lfm-moe/cuda-f16-full/cuda/run-id.json').read_text())['run_id']
rec=Recorder('benchmarks',meta,kind='geometry',parent=parent,
 change='Tune short-prefill expert projection tile geometry after packed half2 codec changes')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
passed=False;compiled=[]
try:
 for rows in (32,512):
  manifest=json.loads((PROGRAM/next(p['path'] for p in index['phases'] if p['phase']=='prefill' and p['chunk']==rows)).read_text())
  for kind,depth,columns in (('gate',2048,1536),('down',1536,2048)):
   record=next(k for k in manifest['kernels'].values() if k['kind']=='quantized_experts' and k['parameters']['k']==depth)
   spec=dict(kind='quantized_experts',parameters=record['parameters'],contract='reference',phase='prefill')
   for bm,bn,bk,threads in choices:
    axes=dict(block_m=bm,columns=bn,block_k=bk,threads=threads)
    implementation,schedule=definition(spec,axes,dict(provider='cuda',target='sm_89'),preferred='akas.ops.moe.cuda:compact_experts')
    name=f'{kind}-r{rows}-{bm}-{bn}-{bk}-{threads}';source=ROOT/(name+'.py')
    source.write_text(selected_source(spec,schedule,dict(provider='cuda',target='sm_89'),implementation).render())
    build_artifact(source,source.with_suffix('.tbin'),provider='cuda',target='sm_89')
    compiled.append(dict(name=name,kind=kind,rows=rows,parameters=record['parameters'],schedule=axes))
    write(ROOT/'compilation.json',compiled);print('BUILT',name,flush=True)
 gguf=GGUF(CHECKPOINT)
 with tensor.Device(provider='cuda') as device:
  for rows in (32,512):
   manifest=json.loads((PROGRAM/next(p['path'] for p in index['phases'] if p['phase']=='prefill' and p['chunk']==rows)).read_text())
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
     variants=[]
     for item in compiled:
      if item['kind']!=kind or item['rows']!=rows:continue
      module=device.load(ROOT/(item['name']+'.tbin'));owned.callback(module.release);variants.append((item['name'],module))
     for layout in ('balanced','skewed'):
      if layout=='balanced':ids=(np.arange(rows)[:,None]*17+np.arange(4)[None,:]*13)%64
      else:ids=np.column_stack((np.zeros(rows,dtype='int32'),np.ones(rows,dtype='int32'),2+np.arange(rows)%62,2+(np.arange(rows)+17)%62))
      arrays['counts'].fill(0);arrays['routes'].fill(-1);sampled=[]
      for expert in range(64):
       owners,ranks=np.where(ids[:-1]==expert)
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
      for geometry,module in [('original-before',baseline),*variants,('original-after',baseline)]:
       transfer(device,buffers['out'],arrays['out']);args=tuple(buffers[a['name']] for a in module.manifest['arguments'])
       plan=ExecutionPlan(device);plan.bind(module,args);graph=plan.capture(resources=tuple(buffers.values())+(module,))
       try:
        graph.launch();actual=buffers['out'].to_numpy()
        for owner,rank,selected,independent in sampled:np.testing.assert_allclose(actual[owner,rank,selected],independent,rtol=.002,atol=.0002)
        if expected is None:expected=actual.copy()
        np.testing.assert_array_equal(actual.view('uint32'),expected.view('uint32'))
        np.testing.assert_array_equal(actual[-1],arrays['out'][-1])
        timing=measure(device,graph.launch,warmup=3,samples=7,repeats=3)
       finally:graph.close()
       check=dict(rows=rows,kind=kind,layout=layout,geometry=geometry,bitwise=True,
        independent_reference_rows=len(sampled),independent_columns=33,inactive_unchanged=True,timing=timing)
       meta.setdefault('checks',[]).append(check);write(ROOT/'report.json',report)
       print('PASS',rows,kind,layout,geometry,timing['median_seconds']*1000,flush=True)
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 rec.artifact(__file__,'kernels/runner.py')
 if (ROOT/'compilation.json').exists():rec.artifact(ROOT/'compilation.json','kernels/compilation.json')
 for path in ROOT.glob('*.tbin'):rec.artifact(path,'kernels/'+path.name)
 write(ROOT/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
 print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

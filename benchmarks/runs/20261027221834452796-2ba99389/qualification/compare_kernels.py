"""Retain bitwise compatibility against the pre-LFM3 CUDA implementations."""
import json, subprocess
from pathlib import Path
import numpy as np
import tensor, akas
from akas.graph import Graph
from akas.kernels.contracts import Implementation
from akas.kernels.registry import validation
from akas.compilation.compile import produce
from akas.compilation.tuning import SearchSettings
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder, write
root=Path('build/lfm3/compat');root.mkdir(parents=True,exist_ok=True)
akas.__path__.append(str(Path('build').resolve()))
for name,path in [('before_conv','src/akas/ops/convolution/kernel.py'),('before_linear','src/akas/ops/linear/cuda.py')]:
 source=subprocess.check_output(['git','show','HEAD:'+path],text=True)
 if name=='before_linear':source=source.replace('from .kernel','from akas.ops.linear.kernel').replace('from .definition','from akas.ops.linear.definition')
 Path('build/lfm3',name+'.py').write_text(source)
Path('build/lfm3/compat_factory.py').write_text('''def before(spec,schedule):
 from akas.lfm3.before_linear import _packed
 return _packed({**spec.parameters,**schedule},paired=spec.kind=="ffn")
def after(spec,schedule):
 from akas.ops.linear.cuda import _packed
 return _packed({**spec.parameters,**schedule},paired=spec.kind=="ffn")
''')
meta=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',checks=[])
rec=Recorder('benchmarks',meta,kind='qualification',change='LFM3 packed projection loads and register convolution bitwise compatibility')
try:
 with tensor.Device() as device:
  for tag,kind,p,half in [
   ('conv-1','conv',dict(r=2,c=128,length=7,chunk=1),False),
   ('conv-32','conv',dict(r=2,c=128,length=7,chunk=32),False),
   ('conv-512','conv',dict(r=2,c=128,length=7,chunk=512),False),
   ('linear-f32','linear',dict(r=32,k=256,o=128,type=2,ggml=True),False),
   ('ffn-f32','ffn',dict(r=32,k=256,o=128,type=2,ggml=True),False),
   ('linear-f16-tail','linear',dict(r=129,k=256,o=128,type=2,ggml=True),True),
   ('ffn-f16-tail','ffn',dict(r=129,k=256,o=128,type=2,ggml=True),True),
  ]:
   paired=kind=='ffn'
   if kind=='conv':
    r,c,L,b=p['r'],p['c'],p['length'],p['chunk']
    specs=[('x',(r*b*3*c,),'float32'),('w',(c*L,),'float32'),('state',(r,L-1,c),'float32'),('out',(r*b*c,),'float32'),('enabled',(r,),'int32')]
   else:
    r,k,o=p['r'],p['k'],p['o']
    specs=[('x',(r*k,),'float32'),('w',(o*k//32*18,),'uint8')]+([('w2',(o*k//32*18,),'uint8')] if paired else [])+[('out',(r*o,),'float32')]
   sig=[dict(name=n,shape=list(shape),dtype=dtype) for n,shape,dtype in specs]
   initial=validation(kind,p,sig).arguments
   if kind=='conv':initial['enabled'][:]=[max(1,p['chunk']-1),0]
   if half:initial['x']=initial['x'].astype('float16').astype('float32')
   outputs=[]
   for version in ('before','after'):
    pp={**p};ss=list(specs)
    if half and version=='after':pp['input_dtype']='float16';ss[0]=('x',ss[0][1],'float16')
    g=Graph(contract='reference')
    args=[g.buffer(n,shape,dtype,role='output' if n in ('out','state') else 'weight' if n.startswith('w') else 'input') for n,shape,dtype in ss]
    g(kind,pp,*args)
    if kind=='conv':
     module='akas.lfm3.before_conv' if version=='before' else 'akas.ops.convolution.kernel'
     factory='convolution';defaults=()
    else:
     module='akas.lfm3.compat_factory';factory=version
     defaults=(('block_m',32),('block_n',64),('block_k',64),('stages',2),('threads',128))
    dest=root/(tag+'-'+version)
    choice=Implementation(kind,module,'reference',factory=factory,defaults=defaults,name='compat.'+version)
    produce(g,dest,device=device,catalog=(choice,),search_settings=SearchSettings(budget=1,samples=1,repeats=1))
    buffers={n:device.from_numpy(initial[n].astype(dtype)) for n,_,dtype in ss}
    try:
     with Forward(dest,device,buffers) as forward:forward.run()
     outputs.append({n:buffers[n].to_numpy() for n in ('out','state') if n in buffers})
    finally:
     for b in buffers.values():b.release()
   for name,value in outputs[0].items():np.testing.assert_array_equal(value.view('uint32'),outputs[1][name].view('uint32'))
   meta['checks'].append(dict(case=tag,parameters=p,outputs=list(outputs[0]),bitwise=True))
   print('BITWISE',tag,flush=True)
 for path in [Path(__file__),Path('build/lfm3/before_conv.py'),Path('build/lfm3/before_linear.py'),Path('build/lfm3/compat_factory.py')]:rec.artifact(path,'qualification/'+path.name)
 record=rec.update(dict(metadata=meta,results=[]),final=True,status='passed')
 write(root/'result.json',dict(run_id=rec.id,status=record['status']))
 print(rec.id,record['status'],flush=True)
except BaseException:
 rec.update(dict(metadata=meta,results=[]),final=True,status='failed');raise

"""Compare retained pre-change expert bodies with new default precision boundaries."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import tensor,akas
from akas.benchmarking.history import Recorder
from akas.kernels.definition import KernelSpec
from akas.kernels.registry import validation
from tensor.compiler.frontend.signature import frontend_arguments

main=Path('/home/sagemaker-user/akas')
root=main/'build/lfm-serving/precision_compat'
root.mkdir(parents=True,exist_ok=True)
old=main/'build/worktrees/expert-pipeline/src/akas/ops/moe/cuda.py'
(root/'before_moe.py').write_text(old.read_text().replace('from .','from akas.ops.moe.'))
akas.__path__.append(str(root.parent))
(root/'factory.py').write_text('''def make(kind,p,before):
 from importlib import import_module
 module=import_module("akas.precision_compat.before_moe" if before else "akas.ops.moe.cuda")
 return module._bucketed({**p,"rows":p["r"]},paired=kind!="expert_partial",quantized_output=kind=="fp8_expert_swiglu")
''')
from akas.precision_compat.factory import make
meta=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',checks=[],before_source_sha256=hashlib.sha256(old.read_bytes()).hexdigest())
rec=Recorder(main/'benchmarks',meta,source_root=main,kind='qualification',change='Preserve default expert arithmetic while exposing product and activation storage precision',dependency_sources={'tensor':main.parent/'tensor'})
for path in (Path(__file__),root/'before_moe.py',root/'factory.py'):rec.artifact(path,path.name)
try:
 with tensor.Device() as device:
  for index,(kind,mma,layout) in enumerate([
   ('expert_partial','float8_e4m3fn','wide'),('expert_partial','float16','wide'),
   ('fp8_expert_swiglu','float8_e4m3fn','wide'),('fp8_expert_swiglu','float16','separate'),
   ('expert_swiglu','float8_e4m3fn','separate'),('expert_swiglu','float16','wide')]):
   p=dict(r=33,rows=33,k=384,o=256,experts=7,top=3,partitions=1,block_m=32,columns=128,threads=256,compact=True,tile_map=True,stages=2,activation_vector=16,mma_dtype=mma,pair_layout=layout)
   if kind=='expert_swiglu':p.update(input_dtype='uint8',weight_format='fp8')
   signature=frontend_arguments(make(kind,p,False),{})
   case=validation(kind,p,signature)
   rng=np.random.default_rng(1700+index)
   for name in ('x','w','w2'):
    if name not in case.arguments:continue
    bits=rng.integers(32,81,case.arguments[name].shape,dtype='uint8')
    case.arguments[name][:]=bits | (rng.integers(0,2,bits.shape,dtype='uint8')*128)
   outputs=[]
   for before in (True,False):
    entry=root/f'case-{index}-{before}.py'; artifact=entry.with_suffix('.tbin')
    entry.write_text('from akas.precision_compat.factory import make\ndef tensor_export():\n return {"kernel":make('+repr(kind)+','+repr(p)+','+repr(before)+')}\n')
    if artifact.exists():artifact.unlink()
    tensor.build(entry,artifact,target='sm_89')
    with device.load(artifact) as executable:
     buffers={item['name']:device.from_numpy(case.arguments[item['name']],dtype=item['dtype']) for item in signature}
     executable.bind(*(buffers[item['name']] for item in signature)).launch()
     outputs.append({name:buffers[name].to_numpy() for name in ('out','output_scales') if name in buffers})
     for value in buffers.values():value.release()
    rec.artifact(entry,entry.name);rec.artifact(artifact,artifact.name)
   for name in outputs[0]:np.testing.assert_array_equal(outputs[0][name],outputs[1][name])
   check=dict(case=index,kind=kind,parameters=p,outputs=list(outputs[0]),bitwise=True)
   meta['checks'].append(check);print('BITWISE',index,kind,mma,layout,flush=True)
 rec.update(dict(metadata=meta,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 rec.update(dict(metadata=meta,results=[],error=repr(error)),final=True,status='failed');raise

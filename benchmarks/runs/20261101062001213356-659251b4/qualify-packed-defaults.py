"""Retain default expert/merge arithmetic when unifying split reductions."""
from pathlib import Path
from contextlib import ExitStack
import importlib,json,shutil
import numpy as np,tensor,akas
from akas.benchmarking.history import Recorder
from akas.kernels.definition import KernelSpec
from akas.kernels.registry import validation
from tensor.compiler.frontend.signature import frontend_arguments
main=Path('/home/sagemaker-user/akas');work=Path.cwd();root=work/'build/lfm-serving/packed_expert_compat'
root.mkdir(parents=True,exist_ok=True);akas.__path__.append(str(root.parent))
for name,owner in [('moe','moe'),('linear','linear'),('activation','activation')]:
 path=(work/'build/lfm-serving/before-packed-experts-cuda.py') if name=='moe' else (main/'src/akas/ops/activation/kernel.py') if name=='activation' else main/'src/akas/ops/linear/cuda.py'
 (root/f'before_{name}.py').write_text(path.read_text().replace('from .',f'from akas.ops.{owner}.'))
(root/'factory.py').write_text('''def make(kind,p,before,legacy=False):
 from importlib import import_module
 from akas.kernels.definition import KernelSpec
 owner="activation" if kind in ("swiglu","swiglu_experts") else "linear" if kind=="expert_merge" else "moe"
 module=import_module("akas.packed_expert_compat.before_"+owner if before else "akas.ops."+owner+(".kernel" if owner=="activation" else ".cuda"))
 if owner=="activation":return module._swiglu(p)
 if owner=="linear":return module.fp8(KernelSpec(kind,p,"reference"),{})
 if legacy:return module.partial(KernelSpec(kind,p,"reference"),{})
 return module._bucketed(p,paired=kind!="expert_partial",quantized_output=kind=="fp8_expert_swiglu")
''')
from akas.packed_expert_compat.factory import make
p=dict(r=33,rows=33,k=384,o=256,experts=7,top=3,partitions=1,block_m=32,columns=128,threads=256,compact=True,tile_map=True,max_block_m=128,stages=1,activation_vector=16)
cases=[
 ('expert_partial',{**p,'r':1,'rows':1,'block_m':16,'threads':128,'mma_dtype':'float16','routed_input':True},False),
 ('expert_partial',{**p,'output_dtype':'bfloat16','routed_input':True},False),
 ('fp8_expert_swiglu',{**p,'product_dtype':'bfloat16','activation_dtype':'bfloat16'},False),
 ('expert_swiglu',{**p,'r':4,'rows':4,'block_m':16,'threads':128,'columns':32,'mma_dtype':'float16','input_dtype':'uint8','weight_format':'fp8','product_dtype':'bfloat16','activation_dtype':'bfloat16','output_dtype':'bfloat16'},False),
 ('expert_partial',{**p,'compact':False,'tile_map':False,'partitions':3,'stages':2},True),
 ('expert_partial',{**p,'r':1,'rows':1,'k':512,'experts':64,'top':4,'block_m':16,'columns':64,'threads':128,'compact':False,'tile_map':False,'partitions':4,'stages':2},True),
 ('expert_merge',dict(r=3,top=4,o=256,partitions=4),False),
 ('swiglu',dict(r=4,c=513,flat=True),False),
 ('swiglu_experts',dict(r=4,top=3,c=513,expert_rows=True,input_dtype='bfloat16',output_dtype='bfloat16'),False),
 ('swiglu',dict(r=4,c=513,packed=True),False),
]
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',checks=[])
rec=Recorder(main/'benchmarks',metadata,source_root=work,kind='qualification',change='Preserve existing compact, paired and noncompact split arithmetic through a shared reduction body',dependency_sources={'tensor':main.parent/'tensor'})
for path in (Path(__file__),*root.glob('*.py')):rec.artifact(path,path.name)
print('STARTED',rec.id,flush=True)
try:
 with tensor.Device() as device:
  for index,(kind,p,legacy) in enumerate(cases):
   signature=frontend_arguments(make(kind,p,False,legacy),{})
   case=validation(kind,p,signature)
   outputs=[]
   for before in (True,False):
    assert signature==frontend_arguments(make(kind,p,before,legacy),{})
    entry=root/f'case{index}-before{before}.py';artifact=entry.with_suffix('.tbin')
    entry.write_text('from akas.packed_expert_compat.factory import make\ndef tensor_export():\n return {"kernel":make('+repr(kind)+','+repr(p)+','+repr(before)+','+repr(legacy)+')}\n')
    if artifact.exists():artifact.unlink()
    tensor.build(entry,artifact,target='sm_89')
    rec.artifact(entry,entry.name);rec.artifact(artifact,artifact.name)
    with device.load(artifact) as executable,ExitStack() as buffers:
     values={a['name']:device.from_numpy(case.arguments[a['name']],dtype=a['dtype']) for a in signature}
     for value in values.values():buffers.callback(value.release)
     executable.bind(*(values[a['name']] for a in signature)).launch()
     outputs.append({name:values[name].to_numpy() for name in ('out','output_scales') if name in values})
   for name in outputs[0]:np.testing.assert_array_equal(outputs[0][name],outputs[1][name])
   metadata['checks'].append(dict(case=index,kind=kind,legacy=legacy,parameters=p,bitwise=True))
   print('BITWISE',index,kind,legacy,flush=True)
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

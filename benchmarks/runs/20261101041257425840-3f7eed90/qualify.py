"""Retain unchanged default primitives and independent extended-map execution."""
from pathlib import Path
import importlib,json,hashlib
import numpy as np,tensor,tilelang,tvm,akas
from akas.benchmarking.history import Recorder
from akas.kernels.definition import KernelSpec
from akas.ops.moe.validation import tile_tables

main=Path('/home/sagemaker-user/akas');work=Path.cwd()
root=work/'build/lfm-serving/map_compat';root.mkdir(parents=True,exist_ok=True)
akas.__path__.append(str(root.parent))
for name in ('cuda','kernel','webgpu'):
 (root/f'before_{name}.py').write_text((main/f'src/akas/ops/moe/{name}.py').read_text().replace('from .','from akas.ops.moe.'))
(root/'factory.py').write_text('''def make(p,portable,before):
 from importlib import import_module
 name="kernel" if portable else "cuda"
 module=import_module("akas.map_compat.before_"+name if before else "akas.ops.moe."+name)
 from akas.kernels.definition import KernelSpec
 return module.expert_tile_map(KernelSpec("expert_tile_map",p,"reference"),{})
''')
from akas.map_compat.factory import make
meta=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',checks=[])
rec=Recorder(main/'benchmarks',meta,source_root=work,kind='qualification',change='Preserve default expert primitives and extend the shared map to 128-row tiles',dependency_sources={'tensor':main.parent/'tensor'})
rec.artifact(Path(__file__),'qualify.py');print('STARTED',rec.id,flush=True)
try:
 p=dict(r=33,top=3,experts=7,o=128,k=128,weight_parts=({'rows':7*128,'format':'ggml','options':{'encoding':2}},),tile_map=True,block_m=16)
 for owner,factory in [('cuda','compact_experts'),('webgpu','register_experts')]:
  before=importlib.import_module('akas.map_compat.before_'+owner)
  after=importlib.import_module('akas.ops.moe.'+owner)
  old=getattr(before,factory)(KernelSpec('quantized_experts',p,'reference'),{})
  new=getattr(after,factory)(KernelSpec('quantized_experts',p,'reference'),{})
  assert tvm.ir.structural_equal(old,new,map_free_vars=True),(owner,factory)
  meta['checks'].append(dict(case=owner+'.'+factory,bitwise=True,method='identical complete default primitive IR'))
  for label,primitive in [('before',old),('after',new)]:
   path=root/(owner+'-'+label+'.json');path.write_text(tvm.ir.save_json(primitive));rec.artifact(path,path.name)
 with tensor.Device() as device:
  for portable in (False,True):
   p=dict(r=257,experts=7,top=3)
   assert tvm.ir.structural_equal(make(p,portable,True),make(p,portable,False),map_free_vars=True)
   meta['checks'].append(dict(case='portable' if portable else 'cuda',bitwise=True,method='identical complete default map primitive IR'))
   for before,maximum in ((True,64),(False,64),(False,128)):
    params={**p,'max_block_m':maximum} if not before else p
    label=f'portable{portable}-before{before}-max{maximum}'
    entry=root/(label+'.py');artifact=entry.with_suffix('.tbin')
    entry.write_text('from akas.map_compat.factory import make\ndef tensor_export():\n return {"kernel":make('+repr(params)+','+repr(portable)+','+repr(before)+')}\n')
    if artifact.exists():artifact.unlink()
    tensor.build(entry,artifact,target='sm_89');rec.artifact(entry,entry.name);rec.artifact(artifact,artifact.name)
    with device.load(artifact) as executable:
     for index,counts in enumerate((np.zeros(7,'int32'),np.array([257,256,129,129,0,0,0],'int32'),np.array([110,111,111,110,110,110,109],'int32'))):
      with device.from_numpy(counts) as source:
       actual=executable(source)
       expected=tile_tables(counts,actual[0].shape[1],5 if maximum==64 else 6)
       for value,array in zip(actual,expected):np.testing.assert_array_equal(value.to_numpy(),array);value.release()
      meta['checks'].append(dict(case=label+'-counts'+str(index),bitwise=True,method='exact independent route-map equations'))
    print('BITWISE',label,flush=True)
 for path in root.glob('*.py'):rec.artifact(path,path.name)
 rec.update(dict(metadata=meta,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=meta,results=[],error=repr(error)),final=True,status='failed')
 raise

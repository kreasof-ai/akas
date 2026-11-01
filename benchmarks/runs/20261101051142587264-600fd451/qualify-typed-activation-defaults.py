"""Retain the default arithmetic and prove declared typed fusion boundaries."""
from pathlib import Path
from contextlib import ExitStack
import json
import numpy as np,tensor,akas
from akas.benchmarking.history import Recorder
from akas.kernels.registry import validation
from tensor.compiler.frontend.signature import frontend_arguments
main=Path.cwd();root=main/'build/lfm-serving/typed_activation_compat';root.mkdir(parents=True,exist_ok=True)
akas.__path__.append(str(root.parent))
for owner in ('normalization','residual'):
 (root/f'before_{owner}.py').write_text((main/f'build/lfm-serving/typed-activations/before_{owner}.py').read_text())
(root/'factory.py').write_text('''def make(kind,p,before):
 from importlib import import_module
 from akas.kernels.definition import KernelSpec
 owner="residual" if kind=="add" else "normalization"
 module=import_module("akas.typed_activation_compat.before_"+owner if before else "akas.ops."+owner+".kernel")
 return getattr(module,kind)(KernelSpec(kind,p,"reference"),{})
''')
from akas.typed_activation_compat.factory import make
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',checks=[])
rec=Recorder(main/'benchmarks',metadata,source_root=main,kind='qualification',change='Preserve default residual/normalization arithmetic while declaring independent storage types',dependency_sources={'tensor':main.parent/'tensor'})
for path in (Path(__file__),*root.glob('*.py')):rec.artifact(path,path.name)
print('STARTED',rec.id,flush=True)
try:
 with tensor.Device() as device:
  cases=[('add',dict(r=33,c=513)),('rms',dict(r=4,c=2048,eps=1e-5)),('add_rms',dict(r=4,c=2048,eps=1e-5,separate_residual=True))]
  for kind in ('rms','add_rms'):
   for staged in (False,True):
    cases.append((kind,dict(r=33,c=513,eps=1e-5,algorithm='serial-tree',stage_values=staged)))
  for index,(kind,p) in enumerate(cases):
   signature=frontend_arguments(make(kind,p,False),{});case=validation(kind,p,signature)
   outputs=[]
   for before in (True,False):
    assert signature==frontend_arguments(make(kind,p,before),{})
    entry=root/f'case{index}-before{before}.py';artifact=entry.with_suffix('.tbin')
    entry.write_text('from akas.typed_activation_compat.factory import make\ndef tensor_export():\n return {"kernel":make('+repr(kind)+','+repr(p)+','+repr(before)+')}\n')
    tensor.build(entry,artifact,target='sm_89');rec.artifact(entry,entry.name);rec.artifact(artifact,artifact.name)
    with device.load(artifact) as executable,ExitStack() as buffers:
     values={a['name']:device.from_numpy(case.arguments[a['name']],dtype=a['dtype']) for a in signature}
     for value in values.values():buffers.callback(value.release)
     executable.bind(*(values[a['name']] for a in signature)).launch()
     outputs.append({name:values[name].to_numpy() for name in ('out','residual') if name in values})
   for name in outputs[0]:np.testing.assert_array_equal(outputs[0][name],outputs[1][name])
   metadata['checks'].append(dict(case=index,kind=kind,parameters=p,bitwise=True));print('BITWISE',index,kind,flush=True)
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

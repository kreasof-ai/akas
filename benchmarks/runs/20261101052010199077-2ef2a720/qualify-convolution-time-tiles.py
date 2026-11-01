"""Compare the tiled parallel history implementation to its original kernel."""
from pathlib import Path
from contextlib import ExitStack
import json
import numpy as np,tensor,akas
from akas.benchmarking.history import Recorder
from akas.kernels.registry import validation
from tensor.compiler.frontend.signature import frontend_arguments
main=Path.cwd();root=main/'build/lfm-serving/convolution_tile_compat';root.mkdir(parents=True,exist_ok=True)
akas.__path__.append(str(root.parent))
(root/'before.py').write_text((main/'build/lfm-serving/before-convolution-time-tiles.py').read_text().replace('from .','from akas.ops.convolution.'))
(root/'factory.py').write_text('''def make(p,before):
 from importlib import import_module
 from akas.kernels.definition import KernelSpec
 module=import_module("akas.convolution_tile_compat.before" if before else "akas.ops.convolution.cuda")
 return module.parallel_prefill(KernelSpec("conv",p,"reference"),{})
''')
from akas.convolution_tile_compat.factory import make
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',checks=[])
rec=Recorder(main/'benchmarks',metadata,source_root=main,kind='qualification',change='Preserve convolution arithmetic and history ownership while tiling time within each channel block',dependency_sources={'tensor':main.parent/'tensor'})
for path in (Path(__file__),*root.glob('*.py')):rec.artifact(path,path.name)
print('STARTED',rec.id,flush=True)
try:
 with tensor.Device() as device:
  cases=[dict(r=3,c=513,chunk=65,length=5,columns=32,threads=128),dict(r=4,c=2048,chunk=2048,length=3,columns=8,threads=256)]
  for index,p in enumerate(cases):
   signature=frontend_arguments(make(p,False),{});case=validation('conv',p,signature)
   outputs=[]
   for version in ('before','default','tiled'):
    params=dict(p)
    if version=='tiled':params.update(time_tile=32,columns=32)
    entry=root/f'case{index}-{version}.py';artifact=entry.with_suffix('.tbin')
    entry.write_text('from akas.convolution_tile_compat.factory import make\ndef tensor_export():\n return {"kernel":make('+repr(params)+','+repr(version=='before')+')}\n')
    tensor.build(entry,artifact,target='sm_89');rec.artifact(entry,entry.name);rec.artifact(artifact,artifact.name)
    with device.load(artifact) as executable,ExitStack() as lifetime:
     values={a['name']:device.from_numpy(case.arguments[a['name']],dtype=a['dtype']) for a in signature}
     for value in values.values():lifetime.callback(value.release)
     call=executable.bind(*(values[a['name']] for a in signature))
     repeated=[]
     for repeat in range(3):
      call.launch();repeated.append({name:values[name].to_numpy() for name in ('out','state')})
     outputs.append(repeated)
   for after in outputs[1:]:
    for old,new in zip(outputs[0],after):
     for name in old:np.testing.assert_array_equal(old[name],new[name])
   metadata['checks'].append(dict(case=index,kind='conv',parameters=p,versions=['default','time_tile32'],repeated_calls=3,bitwise=True));print('BITWISE',index,flush=True)
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

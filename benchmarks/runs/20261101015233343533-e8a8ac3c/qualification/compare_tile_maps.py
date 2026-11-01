"""Retain logical route-map equivalence across tighter physical capacities."""
from pathlib import Path
import json,subprocess
import numpy as np
import tensor,akas
from akas.benchmarking.history import Recorder
from akas.kernels.definition import KernelSpec
from akas.ops.moe.validation import tiles
from tensor.compiler.frontend.signature import frontend_arguments
root=Path('build/lfm-serving/qualification').resolve()
akas.__path__.append(str(root.parent))
(root/'before_moe.py').write_text(subprocess.check_output(['git','show','HEAD:src/akas/ops/moe/cuda.py'],text=True).replace('from .','from akas.ops.moe.'))
from akas.qualification.before_moe import tile_map as before
from akas.ops.moe.cuda import tile_map as after
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',checks=[],qualification_scope='Logical route maps and inactive physical padding; no throughput claim')
rec=Recorder('benchmarks',metadata,kind='qualification',change='Qualify shared prefix-table tile emission and tighter request-derived capacities',dependency_sources={'tensor':Path('../tensor')})
rec.artifact(Path(__file__),'qualification/compare_tile_maps.py')
print('STARTED',rec.id,flush=True)
outputs={}
try:
 with tensor.Device() as device:
  for index,(rows,block,empty) in enumerate(((1,16,False),(33,32,False),(257,64,False),(33,16,True))):
   p=dict(r=rows,rows=rows,block_m=block,top=8,experts=256)
   rng=np.random.default_rng(971+rows)
   counts=np.zeros(256,dtype='int32')
   if not empty:
    for row in range(rows-(rows>1)):
     counts[rng.choice(256,8,replace=False)]+=1
   values=[]
   for label,factory,module in (('before',before,'akas.qualification.before_moe'),('after',after,'akas.ops.moe.cuda')):
    primitive=factory(KernelSpec('expert_tiles',p,'block-fp8'),{})
    signature=frontend_arguments(primitive,{})
    source=root/f'tile-map-{index}-{label}.py'; artifact=source.with_suffix('.tbin')
    source.write_text(f'from akas.kernels.definition import KernelSpec\nfrom {module} import tile_map\ndef tensor_export():\n return dict(kernel=tile_map(KernelSpec("expert_tiles",{p!r},"block-fp8"),{{}}))\n')
    tensor.build(source,artifact,target='sm_89')
    buffers=[device.from_numpy(counts) if a['name']=='counts' else device.zeros(a['shape'],a['dtype']) for a in signature]
    try:
     with device.load(artifact) as executable:
      executable.launch(*buffers)
     actual=[buffers[i].to_numpy() for i,a in enumerate(signature) if a['name']!='counts']
     expected=tiles(counts,block,actual[0].size)
     for array,target in zip(actual,expected):np.testing.assert_array_equal(array,target)
     values.append(actual)
     for name,array in zip(('experts','offsets'),actual):outputs[f'case{index}_{label}_{name}']=array
    finally:
     for buffer in buffers:buffer.release()
    rec.artifact(source,'qualification/'+source.name)
   for original,updated in zip(*values):
    np.testing.assert_array_equal(original[:updated.size],updated)
   metadata['checks'].append(dict(rows=rows,block_m=block,empty=empty,bitwise=True,independent_reference=True,old_capacity=values[0][0].size,new_capacity=values[1][0].size))
   print('BITWISE',metadata['checks'][-1],flush=True)
 np.savez(root/'tile-map-outputs.npz',**outputs)
 rec.artifact(root/'tile-map-outputs.npz','tile-map-outputs.npz')
 rec.artifact(root/'before_moe.py','qualification/before_moe.py')
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

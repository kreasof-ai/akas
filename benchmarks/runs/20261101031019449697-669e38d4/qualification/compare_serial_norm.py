"""Bitwise saved-source qualification for shared serial normalization staging."""
from pathlib import Path
import subprocess,numpy as np
import tensor,akas
from akas.graph import Graph
from akas.kernels.definition import KernelSpec
from akas.kernels.contracts import Implementation
from akas.kernels.registry import validation
from akas.compilation.compile import produce
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder
from tensor.compiler.frontend.signature import frontend_arguments
root=Path('build/lfm-serving/qualification').resolve();akas.__path__.append(str(root.parent))
(root/'before_normalization.py').write_text(subprocess.check_output(['git','show','HEAD:src/akas/ops/normalization/kernel.py'],text=True))
(root/'serial_norm_factory.py').write_text('''from importlib import import_module
def before(spec,schedule):return import_module("akas.qualification.before_normalization")._serial({**spec.parameters,**schedule})
def after(spec,schedule):return import_module("akas.ops.normalization.kernel")._serial({**spec.parameters,**schedule})
''')
from akas.qualification.serial_norm_factory import before,after
metadata=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',checks=[],qualification_scope='Default and staged serial RMS results versus saved source; no throughput claim')
rec=Recorder('/home/sagemaker-user/akas/benchmarks',metadata,kind='qualification',change='Retain serial reduction arithmetic while sharing fused normalization and staging row values',dependency_sources={'tensor':Path('/home/sagemaker-user/tensor')})
print('STARTED',rec.id,flush=True)
try:
 with tensor.Device() as device:
  for rows,width,staged in ((1,256,False),(4,2048,False),(33,513,False),(1,256,True),(4,2048,True),(33,513,True)):
   p=dict(r=rows,c=width,eps=1e-6,stage_values=staged);case_id=f'r{rows}-c{width}-staged{staged}';results=[]
   signature=frontend_arguments(after(KernelSpec('rms',p,'reference'),{}),{})
   case=validation('rms',p,signature)
   for label in ('before','after'):
    graph=Graph(contract='reference')
    bindings={a['name']:graph.buffer(a['name'],a['shape'],a['dtype'],role='output' if a['name']=='out' else 'input') for a in signature}
    graph('rms',p,**bindings)
    choice=Implementation('rms','akas.qualification.serial_norm_factory','reference',factory=label,name='normalization.compat.'+label)
    destination=root/(case_id+'-'+label);produce(graph,destination,search=False,catalog=(choice,),target='sm_89')
    buffers={a['name']:device.from_numpy(case.arguments[a['name']]) for a in signature}
    try:
     with Forward(destination,device,buffers) as forward:forward.run()
     actual=buffers['out'].to_numpy();np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
     results.append(actual)
    finally:
     for value in buffers.values():value.release()
   np.testing.assert_array_equal(results[0].view('uint32'),results[1].view('uint32'))
   metadata['checks'].append(dict(case=case_id,bitwise=True,independent_reference=True));print('BITWISE',case_id,flush=True)
 for path in (Path(__file__),root/'before_normalization.py',root/'serial_norm_factory.py'):rec.artifact(path,'qualification/'+path.name)
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

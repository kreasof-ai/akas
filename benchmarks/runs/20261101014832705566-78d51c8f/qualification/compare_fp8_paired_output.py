"""Retain bitwise old/new evidence using the existing kernel qualification workflow."""
from pathlib import Path
import json,subprocess
import numpy as np
import tensor,akas
from akas.graph import Graph
from akas.kernels.contracts import Implementation
from akas.kernels.registry import validation
from akas.kernels.definition import KernelSpec
from akas.compilation.compile import produce
from akas.execution.forward import Forward
from akas.benchmarking.history import Recorder,write
from tensor.compiler.frontend.signature import frontend_arguments
root=Path('build/lfm-serving/qualification').resolve()
akas.__path__.append(str(root.parent))
for name in ('linear','moe'):
 source=subprocess.check_output(['git','show',f'HEAD:src/akas/ops/{name}/cuda.py'],text=True)
 (root/f'before_{name}.py').write_text(source.replace('from .',f'from akas.ops.{name}.'))
(root/'compat_factory.py').write_text('''def make(spec,schedule,before):
 from importlib import import_module
 owner="moe" if spec.kind=="expert_partial" else "linear"
 module=import_module("akas.qualification.before_"+owner if before else "akas.ops."+owner+".cuda")
 p={**spec.parameters,**schedule}
 return module._bucketed({**p,"rows":p["r"]}) if owner=="moe" else module._mma(p,grouped=False,prequantized=spec.kind=="projection_partial")
def before(spec,schedule):return make(spec,schedule,True)
def after(spec,schedule):return make(spec,schedule,False)
''')
from akas.qualification.compat_factory import after,before
meta=dict(runtime='akas',purpose='kernel-qualification',provider='cuda',target='sm_89',baseline_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checks=[])
rec=Recorder('benchmarks',meta,kind='qualification',change='Preserve default FP8 arithmetic while sharing paired SwiGLU products with a separate output quantizer', dependency_sources={'tensor':Path('../tensor')})
cases=[
 ('dense-inline-tail','fp8_linear',dict(r=17,k=256,o=128,partitions=1,block_m=16,columns=64,threads=128,stages=1)),
 ('dense-prequantized-tail','projection_partial',dict(r=33,k=256,o=128,partitions=1,block_m=32,columns=128,threads=128,stages=2)),
 ('dense-prequantized-split','projection_partial',dict(r=17,k=512,o=128,partitions=2,block_m=16,columns=64,threads=128,stages=1)),
 ('expert-one-row-compact','expert_partial',dict(r=1,k=256,o=128,experts=256,top=8,partitions=1,block_m=16,columns=64,threads=128,compact=True,stages=1,activation_vector=16)),
 ('expert-ragged','expert_partial',dict(r=3,k=256,o=128,experts=256,top=8,partitions=1,block_m=16,columns=64,threads=128,compact=False,stages=1,activation_vector=1)),
 ('expert-routed','expert_partial',dict(r=3,k=256,o=128,experts=256,top=8,partitions=1,block_m=16,columns=64,threads=128,compact=False,stages=1,routed_input=True,activation_vector=16)),
 ('expert-vector-staged','expert_partial',dict(r=33,k=384,o=128,experts=256,top=8,partitions=1,block_m=32,columns=128,threads=256,compact=True,stages=2,activation_vector=16)),
 ('expert-vector-routed','expert_partial',dict(r=33,k=384,o=128,experts=256,top=8,partitions=1,block_m=32,columns=128,threads=256,compact=True,stages=2,routed_input=True,activation_vector=16)),
]
try:
 with tensor.Device() as device:
  for tag,kind,p in cases:
   spec=KernelSpec(kind,p,'block-fp8')
   new_signature=frontend_arguments(after(spec,{}),{})
   case=validation(kind,p,new_signature)
   initial=case.arguments
   outputs=[]
   for version,factory in (('before',before),('after',after)):
    pp=dict(p)
    if version=='before' and p.get('r')==1 and kind=='expert_partial':pp['compact']=False
    primitive=factory(KernelSpec(kind,pp,'block-fp8'),{})
    signature=frontend_arguments(primitive,{})
    graph=Graph(contract='block-fp8')
    args={item['name']:graph.buffer(item['name'],item['shape'],item['dtype'],role='output' if item['name']=='out' else 'input') for item in signature}
    graph(kind,pp,**args)
    choice=Implementation(kind,'akas.qualification.compat_factory','block-fp8',factory=version,name='compat.'+version)
    destination=root/(tag+'-'+version)
    produce(graph,destination,search=False,catalog=(choice,))
    buffers={item['name']:device.from_numpy(initial[item['name']],dtype=item['dtype']) for item in signature}
    try:
     with Forward(destination,device,buffers) as forward:forward.run()
     actual=buffers['out'].to_numpy()
     np.testing.assert_allclose(actual,case.expected['out'],rtol=case.rtol,atol=case.atol)
     outputs.append(actual)
    finally:
     for value in buffers.values():value.release()
   np.testing.assert_array_equal(outputs[0].view('uint32'),outputs[1].view('uint32'))
   meta['checks'].append(dict(case=tag,parameters=p,outputs=['out'],bitwise=True,independent_reference=True))
   print('BITWISE',tag,flush=True)
 for path in (Path(__file__),root/'before_linear.py',root/'before_moe.py',root/'compat_factory.py'):
  rec.artifact(path,'qualification/'+path.name)
 record=rec.update(dict(metadata=meta,results=[]),final=True,status='passed')
 write(root/'result.json',dict(run_id=rec.id,status=record['status']))
 print(rec.id,record['status'],flush=True)
except BaseException:
 rec.update(dict(metadata=meta,results=[]),final=True,status='failed')
 raise

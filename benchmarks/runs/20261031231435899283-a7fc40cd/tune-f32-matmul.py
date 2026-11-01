from pathlib import Path
import json
from akas.graph import Graph
from akas.compilation.compile import plan
from akas.compilation.tuning import qualify,SearchSettings
from akas.benchmarking.history import Recorder

root=Path('build/lfm-serving/f32-matmul-tuning')
root.mkdir(parents=True,exist_ok=True)
graph=Graph(contract='reference',phase='prefill')
for rows in (2048,8192):
 x=graph.buffer(f'x{rows}',(rows,2048),'float32',role='input')
 w=graph.buffer(f'w{rows}',(64,2048),'float32',role='weight')
 out=graph.buffer(f'out{rows}',(rows,64),'float32',role='output')
 graph('matmul',dict(r=rows,k=2048,o=64,dtype='float32',reduction_lanes=32),x=x,w=w,out=out)
record=Recorder('benchmarks',dict(runtime='akas',purpose='kernel-qualification',provider='cuda',
 protocol='operation-tuning-v1',qualification_scope='FP32 matrix projection; not model throughput'),
 kind='benchmark',change='Qualify canonical FP32 matmul schedules with fixed 32-lane reduction')
try:
 profile,report=qualify(plan(graph,target='sm_89'),root,
  settings=SearchSettings(budget=12,width=4,warmup=2,samples=5,repeats=3))
 (root/'profile.json').write_text(json.dumps(report['profile'],indent=2)+'\n')
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 record.artifact(Path(__file__),'tune-f32-matmul.py')
 record.artifact(root/'profile.json','profile.json')
 record.artifact(root/'report.json','qualification.json')
 record.artifact(Path('build/lfm-serving/f32-canonical-bitwise-tests-final.log'),'bitwise-validation.log')
 record.update(dict(results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
 for row in report['operations']:print(row['parameters'],row['selected'],flush=True)
except BaseException:
 record.update(dict(results=[]),final=True,status='failed')
 raise

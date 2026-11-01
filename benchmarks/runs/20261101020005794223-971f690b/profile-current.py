from pathlib import Path
import argparse,json
from akas.benchmarking.history import Recorder
from tools.cuda_profile import profile

root=Path('build/lfm-serving')
parser=argparse.ArgumentParser()
parser.add_argument('--cases',default='[{"batch":1,"chunk":2048,"position":0},{"batch":4,"chunk":1,"position":2048},{"batch":4,"chunk":2048,"position":0}]')
parser.add_argument('--out',type=Path,default=root/'current-captured-profile.json')
args=parser.parse_args()
cases=args.out.with_name(args.out.stem+'-cases.json')
cases.write_text(json.dumps(json.loads(args.cases))+'\n')
metadata=dict(runtime='akas',purpose='kernel-attribution',provider='cuda',protocol='cuda-cupti-activities-v1',
 model='LiquidAI/LFM2.5-24B-A2B-FP8',kv_bits=16,qualification_scope='Captured phase attribution and untimed routing observations; not model throughput')
record=Recorder('benchmarks',metadata,kind='benchmark',change='Locate serving phase bottlenecks and retain actual expert counts',dependency_sources={'tensor':Path('../tensor')})
record.artifact(Path(__file__),'profile-current.py')
record.artifact(cases,'cases.json')
print('STARTED',record.id,flush=True)
try:
 profile(argparse.Namespace(checkpoint=root.parent/'checkpoints/lfm2.5-24b-a2b-fp8',
  program=root/'program-concurrent-fused',definition=None,batch=1,chunk=1,position=0,
  samples=3,cases=cases,captured=True,counter_range=False,observe=['quantized_routes:counts'],
  out=args.out))
 record.program(root/'program-concurrent-fused')
 for path in args.out.parent.glob(args.out.stem+'*.json'):
  record.artifact(path,path.name)
  if '.case-' in path.name and '.sample-' not in path.name:
   case=json.loads(path.read_text())
   assert case['observations']
   for observation in case['observations']:
    counts=observation['outputs']['counts']['values'];p=observation['parameters']
    assert sum(counts)==p['r']*p['top']
 record.update(dict(metadata=metadata,results=[]),final=True,status='passed')
 print('RUN',record.id,flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:
  record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

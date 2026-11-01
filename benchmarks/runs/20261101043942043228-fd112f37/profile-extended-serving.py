"""Retain a captured profile of the measured extended serving program."""
from pathlib import Path
import json,runpy,sys
from akas.benchmarking.history import Recorder
main=Path('/home/sagemaker-user/akas');root=main/'build/lfm-serving/extended-serving-profile'
root.mkdir(parents=True,exist_ok=True)
program=main/'build/worktrees/expert-pipeline/build/lfm-serving/program-concurrent-extended'
cases=[dict(batch=b,chunk=t,position=p) for b,t,p in [(1,2048,0),(4,2048,0),(1,1,2048),(4,1,2048),(16,1,2048),(64,1,2048),(256,1,2048)]]
case_file=root/'cases.json';case_file.write_text(json.dumps(cases,indent=2)+'\n')
metadata=dict(runtime='akas',purpose='captured-kernel-attribution',provider='cuda',measurement_run='20261101042910598214-f1012001',qualification_scope='Captured phase kernel timings and routing observations; not serving throughput or hardware saturation')
rec=Recorder(main/'benchmarks',metadata,source_root=main,kind='diagnostic',change='Locate remaining latency in the exact extended serving program',dependency_sources={'tensor':main.parent/'tensor'})
for path in (Path(__file__),main/'tools/cuda_profile.py',case_file):rec.artifact(path,path.name)
print('STARTED',rec.id,flush=True)
try:
 sys.argv=['cuda_profile.py','--checkpoint',str(main/'build/checkpoints/lfm2.5-24b-a2b-fp8'),'--program',str(program),'--cases',str(case_file),'--samples','3','--captured','--observe','quantized_routes:counts','--out',str(root/'profile.json')]
 runpy.run_path(str(main/'tools/cuda_profile.py'),run_name='__main__')
 for path in root.glob('*.json'):rec.artifact(path,path.name)
 rec.program(program)
 rec.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',rec.id,flush=True)
except BaseException as error:
 if rec.record['completed_at'] is None:rec.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

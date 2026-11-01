from pathlib import Path
import json,os,subprocess
from akas.benchmarking.history import Recorder
root=Path('build/lfm-serving');metadata=dict(runtime='sglang',purpose='numerical-baseline',provider='cuda',model='LiquidAI/LFM2.5-24B-A2B-FP8',kv_dtype='bfloat16',qualification_scope='Natural-prompt completions and probabilities; not serving throughput',baseline_run='20261031182740133509-0f8da2dc')
record=Recorder('benchmarks',metadata,kind='diagnostic',change='Calibrate natural-prompt reference differences against the measured SGLang baseline')
for path in (Path(__file__),root/'baseline-text-probes-worker.py',root/'reference-text-suite.json'):record.artifact(path,path.name)
print('STARTED',record.id,flush=True)
try:
 env=dict(os.environ,CUDA_HOME='/home/sagemaker-user/akas/build/baselines/cuda-13.0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MAX_JOBS='2',LD_LIBRARY_PATH='/home/sagemaker-user/akas/build/baselines/cuda-13.0/lib64:/usr/local/cuda/lib64')
 env['PATH']=env['CUDA_HOME']+'/bin:'+env['PATH']
 subprocess.run(['/home/sagemaker-user/akas/build/baselines/sglang/bin/python','-u',str(root/'baseline-text-probes-worker.py')],env=env,check=True)
 record.artifact(root/'sglang-text-probes.json','sglang-text-probes.json');record.update(dict(metadata=metadata,results=[]),final=True,status='passed');print('RUN',record.id,flush=True)
except BaseException as error:
 if record.record['completed_at'] is None:record.update(dict(metadata=metadata,results=[],error=repr(error)),final=True,status='failed')
 raise

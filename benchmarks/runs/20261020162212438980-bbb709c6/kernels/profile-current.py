"""Attribute the qualified FP8 path with ordered graph events."""
import json,time,traceback
from pathlib import Path
import numpy as np
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.session import Session
from akas.benchmarking.history import Recorder
from cuda_profile import Profile,summarize
meta=dict(runtime='akas',purpose='kernel',protocol='ordered-forward-profile-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),model='akas.models.lfm2.model:LFM2MoE',checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf',kv_dtype='fp8',mode='ar',shape_timeout_seconds=120,qualification='Diagnostic ordered events perturb timing; use native/HTTP run for throughput claims. Full-model streaming weights and real routing.',reference_run_id='20261020152431734974-3777f95e')
rec=Recorder('benchmarks',meta,kind='diagnostic',change='Profile the qualified FP8 model before selecting the next manual optimization')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 with tensor.Device(provider='cuda') as device:
  with LFM2MoE(meta['checkpoint'],'build/lfm-moe/codec-model/cuda/program',device,graphs=True) as model:
   rng=np.random.default_rng(17)
   for prefix,chunk in ((32,32),(8192,4096)):
    prompt=rng.integers(0,128000,prefix,dtype='int32').tolist();frame=model.batches.select(1,chunk);profile=Profile(frame.forward);original=frame.forward.run;samples=[]
    def run_profiled():profile.graph.launch();samples.append(profile.read())
    frame.forward.run=run_profiled
    try:
     with Session(model,batch=1,chunk=chunk,timeout=120) as session:
      for _ in range(3):session.reset();session.prefill([prompt],read=False)
    finally:frame.forward.run=original;profile.close()
    result=summarize(profile,samples[1:]);result.update(prefix=prefix,chunk=chunk,samples=len(samples)-1,phase='prefill');meta.setdefault('profiles',[]).append(result);print('PROFILE',json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
   with Session(model,batch=1,chunk=4096,timeout=120) as session:
    session.reset();session.prefill([prompt],read=False);session.decode(4,read=False);frame=model.last_frame;profile=Profile(frame.forward);original=frame.forward.graph;frame.forward.graph=profile.graph;samples=[]
    try:
     for _ in range(12):session.decode(1,read=False);samples.append(profile.read())
    finally:frame.forward.graph=original;profile.close()
    result=summarize(profile,samples[2:]);result.update(prefix=8192,samples=10,phase='decode');meta.setdefault('profiles',[]).append(result);print('PROFILE',json.dumps({k:v for k,v in result.items() if k!='calls'}),flush=True)
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 rec.artifact(__file__,'kernels/profile-current.py');rec.artifact('build/lfm-moe/cuda_profile.py','kernels/cuda_profile.py');rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

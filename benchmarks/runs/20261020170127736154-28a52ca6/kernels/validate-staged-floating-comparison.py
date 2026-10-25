"""Retain bitwise arithmetic evidence for live-query extent selection."""
import json,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from akas.benchmarking.history import Recorder
from akas.kernels.contracts import sample_arguments
from akas.memory.transfer import write
root=Path('build/lfm-moe/staged-floating-comparison')
meta=dict(runtime='akas',purpose='kernel',protocol='bitwise-kernel-equivalence-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),qualification='Original and updated native F16 attention must be bitwise identical where the previous last-query bound is valid. Covers short/long prefixes, partial/inactive requests, and fresh NaN output sentinels. Poisoned future FP8 cache, odd query tails, and slot reuse are independently tested by test_attention_cache_staging.py.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Qualify live-query extent reduction in native pipelined attention');report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
try:
 signature,_=read_artifact(root/'original.tbin');arrays=sample_arguments(signature['arguments']);arrays['ks'][:]=1;arrays['vs'][:]=1
 with tensor.Device(provider='cuda') as device:
  buffers={n:device.from_numpy(a) for n,a in arrays.items()}
  for length in (32,65,511,512):
   arrays['positions'][:]=np.tile(np.arange(length-32,length,dtype='int32'),2);write(device,buffers['positions'],arrays['positions'])
   for active in ('all','partial','inactive'):
    arrays['enabled'][:]=1
    if active=='partial':arrays['enabled'][31:]=0
    elif active=='inactive':arrays['enabled'][:]=0
    write(device,buffers['enabled'],arrays['enabled']);expected=None
    for label in ('original','updated'):
     write(device,buffers['out'],np.full_like(arrays['out'],np.nan))
     with device.load(root/f'{label}.tbin') as module:
      module.launch(*[buffers[a['name']] for a in module.manifest['arguments']]);actual=buffers['out'].to_numpy()
     assert np.isfinite(actual).all()
     if expected is None:expected=actual.copy()
     else:np.testing.assert_array_equal(actual,expected)
    checks.append(dict(length=length,active=active,bitwise=True));print('PASS',length,active,flush=True)
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/compile-staged-floating-comparison.py',root/'original.py',root/'updated.py'):rec.artifact(path,'kernels/'+Path(path).name)
 for path in ('build/lfm-moe/staging-cuda-tests.log','build/lfm-moe/staging-pipeline-cuda-tests.log','build/lfm-moe/staging-webgpu-tests-v2.log'):rec.artifact(path,'qualification/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

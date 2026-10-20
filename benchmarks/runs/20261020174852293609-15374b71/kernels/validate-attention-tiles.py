"""Include transient decoding costs when comparing identical FP8 coefficients."""
import copy,json,traceback,signal
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.kernels.contracts import sample_arguments
from akas.ops.attention.validation import canonical_validation
from akas.quantization._shared.e4m3 import decode_storage
from akas.memory.transfer import write
from akas.execution.plan import ExecutionPlan
from akas.benchmarking.history import Recorder
root=Path('build/lfm-moe/attention-tiles');shadow_root=Path('build/lfm-moe/fp8-shadow');compiled=json.loads((root/'compilation.json').read_text());program=Path('build/lfm-moe/codec-model/cuda/program');top=read_program(program);manifest=json.loads((program/next(r['path'] for r in top['phases'] if r['phase']=='prefill' and r['chunk']==4096)).read_text());record=next(k for k in manifest['kernels'].values() if k['kind']=='attention');baseline=artifact_path(program,record)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',shape_timeout_seconds=120,provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype='fp8',operand_dtype='float16',accumulator_dtype='float32',qualification='FP8 persistent KV unchanged; reuse exactly the same FP16-decoded coefficients in a temporary cache. Includes live-extent reduction, full shadow decoding, and attention in timing. Small independent CPU coefficient and attention expectations, poisoned future entries, inactive slots/tails, and output reset before every candidate. Full current AOT output comparison; no model/HTTP claim.')
rec=Recorder('benchmarks',meta,kind='algorithm',change='Revisit wider query/key tiles after removing repeated FP8 decoding from prefill MMA')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);checks=[];passed=False
def expired(signum,frame):raise TimeoutError("Attention shape exceeded 120 seconds")
signal.signal(signal.SIGALRM,expired)
try:
 with tensor.Device(provider='cuda') as device:
  for label,length in (('small',97),('full',8192),('full',131072)):
   if label=='small':
    signature,_=read_artifact(root/'small-32-64-128.tbin');signature=copy.deepcopy(signature['arguments'])
    for a in signature:
     if a['name'] in ('kc','vc'):a['dtype']='uint8'
    p=dict(r=66,h=8,kh=2,d=64,capacity=4096,chunk=33,kv_dtype='fp8',provider='cuda',prefill_precision='float16',key_rows=64)
    case=canonical_validation('attention',p,signature);arrays=case.arguments;arrays['enabled'][32:]=0;arrays['q'][32:]=np.nan;case.expected['out'].reshape(66,8,64)[32:]=0;expected=case.expected['out'];rtol,atol=case.rtol,case.atol
   else:
    signature,_=read_artifact(baseline);arrays=sample_arguments(signature['arguments']);rng=np.random.default_rng(44);arrays['positions'][:]=np.arange(length-4096,length,dtype='int32');arrays['enabled'][:]=1;arrays['enabled'][-1]=0;arrays['q'][-1]=np.nan;arrays['mapping'][:]=0
    for n in ('ks','vs'):arrays[n][:]=np.abs(arrays[n])+.01
    for n in ('kc','vc'):
     values=rng.integers(0,100,1048573,dtype='uint8');values|=rng.integers(0,2,1048573,dtype='uint8')<<7;arrays[n][:]=np.resize(values,arrays[n].shape)
    expected=None;rtol,atol=.002,.0002
   slots,heads,cap,depth=arrays['kc'].shape;chunk=len(arrays['positions'])//slots;live=[]
   for slot in range(slots):
    enabled=arrays['enabled'][slot*chunk:(slot+1)*chunk]!=0;positions=arrays['positions'][slot*chunk:(slot+1)*chunk];end=int(positions[enabled].max())+1 if enabled.any() else 0;live.append(end)
    for n in ('kc','vc'):arrays[n][slot,:,end:]=127
    for n in ('ks','vs'):arrays[n][slot,:,end:]=np.nan
   buffers={n:device.from_numpy(a) for n,a in arrays.items()};keys=device.empty(arrays['kc'].shape,'float16');values=device.empty(arrays['vc'].shape,'float16');unit=device.from_numpy(np.ones_like(arrays['ks']));lengths=device.empty((slots,),'int32')
   extent=device.load(shadow_root/f'{label}-extent.tbin');shadow=device.load(shadow_root/f'{label}-shadow.tbin')
   paths=([('baseline',baseline)] if label=='full' else [])+[(item['name'],root/(item['name']+'.tbin')) for item in compiled if item['passed'] and item['label']==label]
   for geometry,path in paths:
    write(device,buffers['out'],np.full_like(arrays['out'],np.nan))
    with device.load(path) as attention:
     plan=ExecutionPlan(device)
     if geometry!='baseline':
      plan.bind(extent,tuple(dict(buffers,lengths=lengths)[a['name']] for a in extent.manifest['arguments']))
      plan.bind(shadow,tuple(dict(buffers,lengths=lengths,keys=keys,values=values)[a['name']] for a in shadow.manifest['arguments']))
      bindings=dict(buffers,kc=keys,vc=values,ks=unit,vs=unit)
     else:bindings=buffers
     plan.bind(attention,tuple(bindings[a['name']] for a in attention.manifest['arguments']));prepared=plan.capture(resources=tuple(buffers.values())+(keys,values,unit,lengths,extent,shadow,attention))
     try:
      prepared.launch();actual=buffers['out'].to_numpy()
      if expected is None:expected=actual.copy()
      np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol)
      if geometry!='baseline':
       np.testing.assert_array_equal(lengths.to_numpy(),np.asarray(live,dtype='int32'))
       if label=='small':
        for source,scale,target in (('kc','ks',keys),('vc','vs',values)):
         decoded=np.zeros(arrays[source].shape,dtype='float16')
         for slot,end in enumerate(live):decoded[slot,:,:end]=(decode_storage(arrays[source][slot,:,:end])*arrays[scale][slot,:,:end]).astype('float16')
         np.testing.assert_array_equal(target.to_numpy(),decoded)
      signal.alarm(120)
      try:timing=measure(device,prepared.launch,warmup=2,samples=5,repeats=1)
      finally:signal.alarm(0)
     finally:prepared.close()
     check=dict(case=label,prefix=length,geometry=geometry,correct=True,bitwise=bool(np.array_equal(actual,expected)),max_absolute_error=float(np.max(np.abs(actual-expected))),temporary_bytes=keys.nbytes+values.nbytes+unit.nbytes+lengths.nbytes,timing=timing);checks.append(check);print('PASS',label,length,geometry,'ms',timing['median_seconds']*1000,'max_error',check['max_absolute_error'],flush=True)
   extent.release();shadow.release();keys.release();values.release();unit.release();lengths.release()
   for b in buffers.values():b.release()
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['checks']=checks
 for path in (__file__,'build/lfm-moe/fp8_shadow.py','build/lfm-moe/compile-fp8-shadow.py','build/lfm-moe/compile-attention-tiles.py',root/'compilation.json','src/akas/ops/attention/cuda.py','build/lfm-moe/attention_wide.py'):rec.artifact(path,'kernels/'+Path(path).name)
 rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

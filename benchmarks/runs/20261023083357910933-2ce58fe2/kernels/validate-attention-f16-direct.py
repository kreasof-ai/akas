"""Direct F16 prefill attention: reference math, poisoned future and unchanged inputs."""
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.benchmarking.history import Recorder,write
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.memory.transfer import write as transfer
from akas.ops.attention.validation import canonical_validation

ROOT=Path('build/lfm-moe/attention-f16-direct')
PROGRAM=Path('build/lfm-moe/cuda-f16-full/cuda/program')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',
 device=dict(name='NVIDIA L4',provider='cuda'),kv_dtype='float16',operand_dtype='float16',accumulator_dtype='float32',
 qualification='Native F16 cache attention; zero only the unused tail read by four 64-key async tiles. Every valid cache value is unchanged. Reuse the qualified canonical-order attention factory. Small independent F64 reference, full-size bitwise sanitized current AOT comparison and poisoned inactive queries/future values; kernel-only timings, not model qualification.')
parent=json.loads(Path('build/lfm-moe/cuda-f16-full/cuda/run-id.json').read_text())['run_id']
rec=Recorder('benchmarks',meta,kind='algorithm',parent=parent,
 change='Prepare the unused native F16 tail in place and reuse the qualified ordered attention factory')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);passed=False
try:
 with tensor.Device(provider='cuda') as device:
  for label,length in [('small',97),('medium',512),('full',8192),('full',130944)]:
   artifact=ROOT/(label+'.tbin');sig,_=read_artifact(artifact)
   slots,chunk,h,kh,cap=(2,33,8,2,4096) if label=='small' else (1,512 if label=='medium' else 4096,32,8,131072)
   p=dict(r=slots*chunk,h=h,kh=kh,d=64,capacity=cap,chunk=chunk,kv_dtype='float16',unit_scales=True,provider='cuda',prefill_precision='float16',key_rows=64)
   if label=='small':
    case=canonical_validation('attention',p,sig['arguments']);arrays=case.arguments
    # Keep the independent CPU reference at its original valid positions.
    arrays['enabled'][32:]=0;arrays['q'][32:]=np.nan;case.expected['out'].reshape(66,8,64)[32:]=0
    expected=case.expected['out'];baseline_path=None
   else:
    arrays=sample_arguments(sig['arguments']);rng=np.random.default_rng(44)
    arrays['positions'][:]=np.arange(length-chunk,length,dtype='int32');arrays['enabled'][:]=1;arrays['enabled'][-1]=0;arrays['q'][-1]=np.nan
    for name in ('kc','vc'):arrays[name][:]=np.resize(rng.normal(0,.2,1048573).astype('float16'),arrays[name].shape)
    for name in ('ks','vs'):arrays[name].fill(1)
    top=read_program(PROGRAM);phase=next(x for x in top['phases'] if x['phase']=='prefill' and x['chunk']==chunk)
    m=json.loads((PROGRAM/phase['path']).read_text());baseline_path=artifact_path(PROGRAM,next(x for x in m['kernels'].values() if x['kind']=='attention'))
    expected=None
    arrays['mapping']=np.arange(slots,dtype='int32')
   ends=[]
   for slot in range(slots):
    mask=arrays['enabled'][slot*chunk:(slot+1)*chunk]!=0;pos=arrays['positions'][slot*chunk:(slot+1)*chunk]
    end=int(pos[mask].max())+1 if mask.any() else 0;ends.append(end)
    for name in ('kc','vc'):arrays[name][slot,:,end:]=0
   with ExitStack() as owned:
    buffers={name:device.from_numpy(array) for name,array in arrays.items()}
    for b in buffers.values():owned.callback(b.release)
    candidate=device.load(artifact);owned.callback(candidate.release)
    padding=device.load(ROOT/(label+'-pad.tbin'));owned.callback(padding.release)
    lengths=device.from_numpy(np.asarray(ends,dtype='int32'));owned.callback(lengths.release)
    variants=[]
    if baseline_path is not None:
     baseline=device.load(baseline_path);owned.callback(baseline.release);variants.append(('baseline',baseline))
    variants.append(('prepared-f16',candidate))
    for name,module in variants:
     if name=='prepared-f16':
      for cache in ('kc','vc'):
       for slot,end in enumerate(ends):arrays[cache][slot,:,end:]=np.nan
       transfer(device,buffers[cache],arrays[cache])
     transfer(device,buffers['out'],np.full_like(arrays['out'],np.nan))
     args=tuple(buffers[a['name']] for a in module.manifest['arguments']);plan=ExecutionPlan(device)
     if name=='prepared-f16':
      pack=dict(buffers,lengths=lengths)
      plan.bind(padding,tuple(pack[a['name']] for a in padding.manifest['arguments']))
     plan.bind(module,args)
     graph=plan.capture(resources=tuple(buffers.values())+(module,))
     try:
      graph.launch();actual=buffers['out'].to_numpy();assert np.isfinite(actual).all()
      if expected is None:expected=actual.copy()
      np.testing.assert_allclose(actual,expected,rtol=.002,atol=.0002)
      if label!='small':np.testing.assert_array_equal(actual.view('uint32'),expected.view('uint32'))
      timing=measure(device,graph.launch,warmup=2,samples=7,repeats=1)
     finally:graph.close()
     for cache in ('kc','vc'):
      if name=='prepared-f16':
       wanted=arrays[cache].copy()
       for slot,end in enumerate(ends):wanted[slot,:,end:min(cap,(end+255)//256*256)]=0
       np.testing.assert_array_equal(buffers[cache].to_numpy(),wanted)
     check=dict(case=label,prefix=length,geometry=name,bitwise=bool(np.array_equal(actual,expected)),independent_f64=label=='small',finite=True,valid_cache_prefix_unchanged=True,unused_future_tail_zeroed=True,timing=timing)
     meta.setdefault('checks',[]).append(check);write(ROOT/'report.json',report)
     print('PASS',label,length,name,timing['median_seconds']*1000,check['bitwise'],flush=True)
 passed=True
except BaseException:
 meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 for source in (__file__,'build/lfm-moe/attention_f16_pad.py','build/lfm-moe/compile-attention-f16-direct.py'):rec.artifact(source,'kernels/'+Path(source).name)
 for artifact in ROOT.glob('*.tbin'):rec.artifact(artifact,'kernels/'+artifact.name)
 write(ROOT/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

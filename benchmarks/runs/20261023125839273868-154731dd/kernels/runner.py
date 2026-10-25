"""Qualify short-block projection geometries on actual packed checkpoint weights."""
from contextlib import ExitStack
import json,resource,time,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program,artifact_path
from akas.benchmarking.history import Recorder,write
from akas.checkpoints.gguf import GGUF
from akas.compilation.emit import build_artifact,needs_build
from akas.compilation.selection import selected_source
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.kernels.definition import definition
from akas.quantization.weights.ggml.validation import decode_rows
root=Path('/home/sagemaker-user/akas');out=root/'build/lfm-moe/dspark-markov-geometry';out.mkdir(exist_ok=True)
base=root/'build/lfm-moe/dspark-long-qualification';target=GGUF(root/'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf');draft=GGUF(root/'build/checkpoints/lfm2.5-24b-a2b/dspark-f16.gguf')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',checks=[],qualification='Actual Q4_0, Q6_K and F16 checkpoint matrices; bounded sampled F64 products with established F16 operand rounding, whole-output bitwise comparison against the retained model kernel. CUDA event timing of each complete projection. No model speedup is inferred from kernel timing alone.')
rec=Recorder(root/'benchmarks',meta,kind='geometry',parent='20261023122111836306-aa15eddc',change='Reduce sequential Markov projection latency using the public packed F16 GEMV and bounded independent validation')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[]);write(out/'run-id.json',dict(run_id=rec.id));print('RUN',rec.id,flush=True);passed=False
try:
 with tensor.Device(provider='cuda') as device:
  for label,program,phase,shape,encoding,reader in (('markov',base/'draft','proposal',(1,256,128000),1,draft),):
   index=read_program(program);record=next(p for p in index['phases'] if p['phase']==phase)
   manifest=json.loads((program/record['path']).read_text());rows,depth,columns=shape
   old=next(k for k in manifest['kernels'].values() if k['kind']=='linear' and tuple(k['parameters'].get(n) for n in ('r','k','o'))==shape and k['parameters'].get('type')==encoding)
   parameters=old['parameters'];name=next(n for n,t in reader.tensors.items() if t.shape==(columns,depth) and t.type==encoding)
   print('MATRIX',label,name,flush=True)
   with ExitStack() as owned:
    baseline=device.load(artifact_path(program,old));owned.callback(baseline.release)
    arrays=sample_arguments(baseline.manifest['arguments']);arrays['x'][:]=np.random.default_rng(91).normal(0,.4,arrays['x'].shape).astype('float16').astype('float32');arrays['w'][:]=reader.packed(name).view(arrays['w'].dtype);reader.discard(name)
    buffers={n:device.from_numpy(a) for n,a in arrays.items()}
    for b in buffers.values():owned.callback(b.release)
    chosen=np.unique(np.linspace(0,columns-1,39,dtype=int));reference=np.empty((rows,len(chosen)))
    for j,col in enumerate(chosen):
     w=decode_rows(arrays['w'],encoding,depth,int(col),int(col)+1).astype('float16').astype('float64').reshape(depth)
     reference[:,j]=arrays['x'].reshape(rows,depth).astype('float16').astype('float64')@w
    expected=None
    geometries=[None,(128,1,4),(256,1,4),(128,1,8),(256,1,8)]
    for geometry in geometries:
     if geometry is None:module=baseline;tag='baseline'
     else:
      threads,unroll,width=geometry;axes=dict(threads=threads,unroll=unroll,f16_values=width)
      spec=dict(kind='linear',parameters=parameters,contract='reference',phase=phase);caps=dict(provider='cuda',target='sm_89')
      impl,schedule=definition(spec,axes,caps,preferred='linear.cuda.packed')
      text=selected_source(spec,schedule,caps,impl).render();tag=f't{threads}-u{unroll}-w{width}';path=out/f'{label}-{tag}.py'
      if needs_build(path,path.with_suffix('.tbin'),text,'sm_89',provider='cuda'):
       path.with_suffix('.tbin').unlink(missing_ok=True);path.write_text(text);build_artifact(path,path.with_suffix('.tbin'),target='sm_89',provider='cuda')
      module=device.load(path.with_suffix('.tbin'));owned.callback(module.release)
     plan=ExecutionPlan(device);plan.bind(module,tuple(buffers[a['name']] for a in module.manifest['arguments']));prepared=plan.capture(resources=tuple(buffers.values())+(module,))
     try:
      prepared.launch();actual=buffers['out'].to_numpy().reshape(rows,columns)
      np.testing.assert_allclose(actual[:,chosen],reference,rtol=.002,atol=.002)
      if expected is None:expected=actual.copy()
      bitwise=np.array_equal(actual.view('uint32'),expected.view('uint32'));maxerr=float(np.max(np.abs(actual-expected)))
      timing=measure(device,prepared.launch,warmup=3,samples=5,repeats=5)
     finally:prepared.close()
     check=dict(matrix=label,parameters=parameters,weight_name=name,geometry=geometry,correct=True,bitwise=bitwise,max_abs_error=maxerr,independent_columns=len(chosen),timing=timing)
     meta['checks'].append(check);rec.update(report);write(out/'report.json',report);print('PASS',label,tag,timing['median_seconds']*1e3,'bitwise',bitwise,'error',maxerr,flush=True)
 passed=True
except BaseException as error:meta['error']=f'{type(error).__name__}: {error}';meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
 meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;rec.artifact(__file__,'kernels/runner.py');write(out/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

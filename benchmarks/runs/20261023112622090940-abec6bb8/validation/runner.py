"""Independent native F16 block attention qualification and complete-plan timing."""
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.artifacts.schema import Source
from akas.benchmarking.history import Recorder,write
from akas.compilation.emit import build_artifact, needs_build
from akas.compilation.selection import selected_source
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import Implementation
from akas.kernels.definition import definition

ROOT=Path('/home/sagemaker-user/akas')
OUT=ROOT/'build/lfm-moe/block-attention';OUT.mkdir(exist_ok=True)
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',
    kv_dtype='float16',operand_dtype='float16',accumulator_dtype='float32',checks=[],
    qualification='Independent F64 sampled heads plus complete-output agreement with guarded native F16 attention; future KV NaN poison and inactive queries. Timings include partial and merge. Kernel evidence, not model throughput.')
rec=Recorder(ROOT/'benchmarks',meta,kind='algorithm',parent='20261023100806678163-765f4009',
    change='Partition native F16 attention while sharing KV across speculative query tokens and GQA heads')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
write(OUT/'run-id.json',dict(run_id=rec.id));rec.update(report)
print('RUN',rec.id,flush=True)
passed=False
try:
    paths={}
    for chunk in (9,10):
        p=dict(r=chunk,chunk=chunk,h=32,kh=8,d=64,capacity=131072,kv_dtype='float16',
               unit_scales=True,provider='cuda',optimized=True,splits=64)
        for label,kind in (('baseline','attention'),('partial','attention_partial'),('merge','attention_merge')):
            spec=dict(kind=kind,parameters=p,contract='reference',phase='verify')
            caps=dict(provider='cuda',target='sm_89')
            if label=='partial':
                schedule=dict(key_rows=64,stages=1)
            else:
                implementation,schedule=definition(spec,{'implementation':'attention.cuda.pipelined'} if label=='baseline' else {},caps)
            path=OUT/f'{label}-{chunk}.py'
            text=(Source('block_attention_partial','make',(spec,schedule)).render() if label=='partial' else selected_source(spec,schedule,caps,implementation).render())
            if needs_build(path,path.with_suffix('.tbin'),text,'sm_89',provider='cuda'):
                path.with_suffix('.tbin').unlink(missing_ok=True)
                path.write_text(text)
                build_artifact(path,path.with_suffix('.tbin'),target='sm_89',provider='cuda')
            paths[chunk,label]=path.with_suffix('.tbin')
            print('COMPILED',chunk,label,flush=True)
    with tensor.Device(provider='cuda') as device,ExitStack() as owned:
        modules={key:device.load(path) for key,path in paths.items()}
        for module in modules.values():owned.callback(module.release)
        for chunk in (9,10):
            for live in (37,8192,130944):
                for causal in (True,False):
                    rng=np.random.default_rng(97+chunk+live)
                    q=rng.normal(0,1,(chunk,2048)).astype('float32')
                    kc=np.full((1,8,131072,64),np.nan,dtype='float16')
                    vc=kc.copy()
                    kc[:,:,:live]=rng.normal(0,.5,(1,8,live,64)).astype('float16')
                    vc[:,:,:live]=rng.normal(0,.3,(1,8,live,64)).astype('float16')
                    positions=(np.arange(chunk)+live-chunk if causal else np.full(chunk,live-1)).astype('int32')
                    enabled=np.ones(chunk,'int32');enabled[-1]=0
                    hosts=dict(q=q,kc=kc,vc=vc,ks=np.ones((1,8,131072,1),'float32'),
                               vs=np.ones((1,8,131072,1),'float32'),positions=positions,enabled=enabled)
                    with ExitStack() as case:
                        buffers={name:device.from_numpy(value) for name,value in hosts.items()}
                        partial=device.empty((chunk*32*64*66,),'float32')
                        out=device.empty((chunk*2048,),'float32')
                        for buffer in (*buffers.values(),partial,out):case.callback(buffer.release)
                        baseline=modules[chunk,'baseline']
                        baseline_args={**buffers,'out':out}
                        baseline.launch(*(baseline_args[a['name']] for a in baseline.manifest['arguments']))
                        expected=out.to_numpy().reshape(chunk,32,64).copy()
                        sample=[]
                        for row in (0,chunk-2):
                            for head in (0,31):
                                count=int(positions[row])+1
                                keys=kc[0,head//4,:count].astype('float64')
                                values=vc[0,head//4,:count].astype('float64')
                                query=q.reshape(chunk,32,64)[row,head].astype('float16').astype('float64')
                                scores=keys@query/8
                                probability=np.exp(scores-scores.max());probability/=probability.sum()
                                reference=probability@values
                                np.testing.assert_allclose(expected[row,head],reference,rtol=.003,atol=.0003)
                                sample.append((row,head,reference))
                        plans={}
                        for label in ('baseline','partitioned'):
                            plan=ExecutionPlan(device)
                            for name in (('baseline',) if label=='baseline' else ('partial','merge')):
                                module=modules[chunk,name]
                                args={**buffers,'out':partial if name=='partial' else out,'parts':partial}
                                plan.bind(module,tuple(args[a['name']] for a in module.manifest['arguments']))
                            prepared=plan.capture(resources=(*modules.values(),*buffers.values(),partial,out))
                            case.callback(prepared.close)
                            prepared.launch()
                            actual=out.to_numpy().reshape(chunk,32,64)
                            np.testing.assert_allclose(actual,expected,rtol=.003,atol=.0003)
                            np.testing.assert_array_equal(actual[-1],0)
                            for row,head,reference in sample:
                                np.testing.assert_allclose(actual[row,head],reference,rtol=.003,atol=.0003)
                            plans[label]=measure(device,prepared.launch,warmup=1,samples=3,repeats=3)
                        for name in ('kc','vc'):
                            np.testing.assert_array_equal(buffers[name].to_numpy().view('uint16'),hosts[name].view('uint16'))
                        check=dict(chunk=chunk,live=live,causal=causal,correct=True,
                            independent_queries=len(sample),timings=plans,
                            speedup=plans['baseline']['median_seconds']/plans['partitioned']['median_seconds'])
                        meta['checks'].append(check);rec.update(report)
                        print('PASS',chunk,live,causal,'speedup',check['speedup'],flush=True)
    passed=True
except BaseException as error:
    meta['error']=f'{type(error).__name__}: {error}'
    meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
    rec.artifact(__file__,'validation/runner.py')
    rec.artifact(Path(__file__).with_name('block_attention_partial.py'),'kernels/factory.py')
    for path in OUT.glob('*.tbin'):rec.artifact(path,'kernels/'+path.name)
    write(OUT/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed')
    print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

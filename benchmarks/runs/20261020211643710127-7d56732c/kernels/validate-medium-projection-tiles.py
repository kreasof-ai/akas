"""Actual packed projection weights, F64 products and full AOT equivalence."""
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from tensor.artifacts.format import read_artifact
from akas.artifacts.read import artifact_path,read_program
from akas.benchmarking.history import Recorder,write
from akas.checkpoints.gguf import GGUF
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.memory.transfer import write as transfer
from akas.quantization.weights.ggml.validation import decode_rows,sample_weight

root=Path('build/lfm-moe/medium-projection-tiles')
program=Path('build/lfm-moe/cuda-full/cuda/program')
compiled=json.loads((root/'compilation.json').read_text())
index=read_program(program)
f=json.loads((program/next(p['path'] for p in index['phases'] if p['phase']=='prefill' and p['chunk']==4096)).read_text())
checkpoint=Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf')
meta=dict(runtime='akas',purpose='kernel',protocol='kernel-latency-v1',provider='cuda',target='sm_89',device=dict(name='NVIDIA L4',provider='cuda'),checkpoint=str(checkpoint),checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',weight_format='q4_0',operand_dtype='float16',accumulator_dtype='float32',reference_run_id='20261020182637613063-8a87f1a5',qualification='Unmodified public CUDA linear factory with larger projection tiles. Actual GGUF weights and synthetic activations; sampled independent F64 products with the established FP16 operand rounding. Entire full-size output must match retained AOT bits. Odd row and output tails checked independently. A/B/candidates/A timing; kernel-only evidence, no model or serving speed claim.')
rec=Recorder('benchmarks',meta,kind='geometry',change='Select projection reuse geometry for unpadded 512- and 2048-token CUDA prefill')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
passed=False
try:
    gguf=GGUF(checkpoint)
    with tensor.Device(provider='cuda') as device:
        for name in sorted({i['case'] for i in compiled}):
            items=[i for i in compiled if i['case']==name]
            p=items[0]['parameters']
            rows,depth,columns=(p[k] for k in ('r','k','o'))
            old=root/(items[0]['name']+'.tbin')
            signature,_=read_artifact(old)
            arrays=sample_arguments(signature['arguments'])
            arrays['x']*=8
            if name=='tails':
                sample_weight(arrays['w'],2,decode=False)
            else:
                suffix={'conv-in':'shortconv.in_proj.weight','projection':'shortconv.out_proj.weight','dense-down':'ffn_down.weight'}[name.split('-r')[0]]
                candidates=sorted(n for n in gguf.tensors if n.endswith(suffix))
                weight=candidates[len(candidates)//2]
                assert gguf.tensors[weight].type==2
                arrays['w'][:]=gguf.packed(weight)
                gguf.discard(weight)
                meta.setdefault('weight_banks',{})[name]=weight
            selected=np.unique(np.linspace(0,rows-1,16,dtype='int32'))
            source=arrays['x'].reshape(rows,depth)[selected].astype('float16').astype('float64')
            reference=np.empty((len(selected),columns),dtype='float32')
            for begin in range(0,columns,128):
                end=min(columns,begin+128)
                weights=decode_rows(arrays['w'],2,depth,begin,end).astype('float16').astype('float64')
                reference[:,begin:end]=source@weights.T
            with ExitStack() as owned:
                buffers={n:device.from_numpy(a) for n,a in arrays.items()}
                for buffer in buffers.values():owned.callback(buffer.release)
                baseline=device.load(old);owned.callback(baseline.release)
                modules={i['name']:device.load(root/(i['name']+'.tbin')) for i in items}
                for module in modules.values():owned.callback(module.release)
                variants=[('original-before',baseline),*modules.items(),('original-after',baseline)]
                expected=None
                for geometry,module in variants:
                    transfer(device,buffers['out'],np.full_like(arrays['out'],np.nan))
                    args=tuple(buffers[a['name']] for a in module.manifest['arguments'])
                    plan=ExecutionPlan(device);plan.bind(module,args)
                    graph=plan.capture(resources=tuple(buffers.values())+(module,))
                    try:
                        graph.launch()
                        actual=buffers['out'].to_numpy().reshape(rows,columns)
                        assert np.isfinite(actual).all()
                        np.testing.assert_allclose(actual[selected],reference,rtol=.002,atol=.0002)
                        if expected is None:expected=actual.copy()
                        np.testing.assert_array_equal(actual.view('uint32'),expected.view('uint32'))
                        timing=measure(device,graph.launch,warmup=3,samples=5,repeats=3)
                    finally:graph.close()
                    check=dict(case=name,parameters=p,geometry=geometry,bitwise=True,independent_reference_rows=len(selected),timing=timing)
                    meta.setdefault('checks',[]).append(check)
                    write(root/'report.json',report)
                    print('PASS',name,geometry,timing['median_seconds']*1000,flush=True)
    passed=True
except BaseException:
    meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
    for path in (Path(__file__),Path('build/lfm-moe/compile-medium-projection-tiles.py'),root/'compilation.json'):
        rec.artifact(path,'kernels/'+path.name)
    for path in sorted(root.glob('*.tbin')):rec.artifact(path,'kernels/'+path.name)
    write(root/'report.json',report)
    rec.update(report,final=True,status='passed' if passed else 'failed')
    print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

"""Separate projection and attention changes using matched real model inputs."""
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.models.lfm2.model import LFM2MoE
from akas.optimization.evaluation import prompts

root=Path('build/lfm-moe/ordered-attention-model');root.mkdir(exist_ok=True)
program=Path('build/lfm-moe/cuda-full/cuda/program')
parent='20261020182637613063-8a87f1a5'
reference=json.loads((Path('benchmarks/runs')/parent/'report.json').read_text())
expected=next(r for r in reference['results'] if r['case']['prompt']==8192)['output_ids'][0]
meta=dict(reference['metadata'])
for name in ('http','precision_comparison','same_precision_logit_comparison','compilation_resources'):
    meta.pop(name,None)
meta.update(purpose='kernel',protocol='model-operation-isolation-v1',requested_prefixes=[8192],
    qualification='Diagnostic separation of larger public projection tiles and wider attention. Actual final-layer Q/K/V compared against an independent float64 block-softmax reference with the established FP16 operand and probability storage. 1 warmup and 3 native repeats; all 128 free-AR IDs compared with the qualified FP8 model. Nine teacher-forced logits diagnose any divergence. A candidate that changes free-AR IDs is not promoted; this is not a serving qualification.',
    kernel_evidence_run_ids=['20261020205941837023-84bc5997','20261020210651016012-a3613aec'],shape_timeout_seconds=120)
rec=Recorder('benchmarks',meta,kind='diagnostic',parent=parent,change='Qualify the warp-local canonical-order attention alone and with larger public projection tiles')
report=dict(schema='akas.benchmark.v1',metadata=meta,results=[])
passed=False
paths={
    'conv-in':Path('build/lfm-moe/projection-tiles/conv-in-128-128-64-256-1.tbin'),
    'projection':Path('build/lfm-moe/projection-tiles/projection-128-128-64-256-1.tbin'),
    'dense-down':Path('build/lfm-moe/projection-tiles/dense-down-64-256-32-256-2.tbin'),
    'square64':Path('build/lfm-moe/attention-tiles/full-64-64-128.tbin'),
    'ordered64':Path('build/lfm-moe/attention-ordered-rows/full.tbin')}
try:
    with tensor.Device(provider='cuda') as device:
        with LFM2MoE(meta['checkpoint'],program,device,graphs=True) as model,ExitStack() as owned:
            print('LOADED',flush=True)
            modules={n:device.load(p) for n,p in paths.items()}
            for module in modules.values():owned.callback(module.release)
            forwards={phase:model.batches.select(1,4096,phase=phase).forward for phase in ('prefill','prefill_state')}
            original={phase:f.graph for phase,f in forwards.items()}
            attention_entry=next((e,c) for e,c in reversed(list(zip(forwards['prefill'].plan,forwards['prefill'].definition.calls))) if c.kind=='attention')
            entry,call=attention_entry
            bindings=call_bindings(entry[0],entry[1])
            # Preserve actual prefill inputs while comparing independent attention stages.
            with Session(model,batch=1,chunk=4096,timeout=120) as session:
                session.reset();session.prefill(prompts(model.config.vocab,1,8192))
                q=bindings['q'].to_numpy().reshape(4096,32,64)
                keys=bindings['kc'].to_numpy();values=bindings['vc'].to_numpy()
                positions=bindings['positions'].to_numpy().reshape(-1)
                active=bindings['enabled'].to_numpy().reshape(-1)!=0
                ks=bindings['ks'].to_numpy();vs=bindings['vs'].to_numpy()
                assert np.all(ks==1) and np.all(vs==1)
                rows=np.flatnonzero(active)
                rows=rows[[0,len(rows)//2,-1]]
                samples=[]
                for row in rows:
                    for head in (0,15,31):
                        query=q[row,head].astype('float16').astype('float64')
                        end=int(positions[row])+1
                        numerator=np.zeros(64,dtype='float64');normalizer=0.;maximum=-np.inf
                        for begin in range(0,end,64):
                            stop=min(end,begin+64)
                            scores=keys[0,head//4,begin:stop].astype('float64')@query/8
                            next_maximum=max(maximum,float(scores.max()))
                            correction=np.exp(maximum-next_maximum)
                            probability=np.exp(scores-next_maximum)
                            numerator=numerator*correction+probability.astype('float16').astype('float64')@values[0,head//4,begin:stop].astype('float64')
                            normalizer=normalizer*correction+probability.sum()
                            maximum=next_maximum
                        samples.append((int(row),head,numerator/normalizer))
                del keys,values,q,ks,vs
                temporary=device.empty(bindings['out'].shape,str(bindings['out'].dtype));owned.callback(temporary.release)
                proof_bindings={**bindings,'out':temporary}
                baseline=None
                for name,module in [('original',entry[0]),('square64',modules['square64']),('ordered64',modules['ordered64'])]:
                    args=tuple(proof_bindings[a['name']] for a in module.manifest['arguments'])
                    plan=ExecutionPlan(device);plan.bind(module,args);graph=plan.capture(resources=tuple(proof_bindings.values())+(module,))
                    try:
                        graph.launch();actual=temporary.to_numpy().reshape(4096,32,64)
                        errors=[]
                        for row,head,reference_value in samples:
                            np.testing.assert_allclose(actual[row,head],reference_value,rtol=.002,atol=.0002)
                            errors.extend((actual[row,head]-reference_value).tolist())
                        timing=measure(device,graph.launch,warmup=3,samples=5,repeats=1)
                    finally:graph.close()
                    if baseline is None:baseline=actual.copy()
                    check=dict(variant=name,independent_f64_passed=True,sampled_rows=len(samples),f64_max_absolute_error=float(np.max(np.abs(errors))),f64_rmse=float(np.sqrt(np.mean(np.square(errors)))),bitwise_original=bool(np.array_equal(actual,baseline)),max_difference_original=float(np.max(np.abs(actual-baseline))),timing=timing)
                    meta.setdefault('real_attention_checks',[]).append(check)
                    print('REAL_ATTENTION',json.dumps(check),flush=True)
                    write(root/'report.json',report)
            candidates={}
            for variant in ('ordered64','projection-ordered64'):
                candidates[variant]={}
                for phase,forward in forwards.items():
                    plan=ExecutionPlan(device,labeled=True)
                    counts=dict(projection=0,attention=0)
                    for old,operation in zip(forward.plan,forward.definition.calls):
                        p=operation.parameters;module=None
                        if variant in ('projection','projection-ordered64') and operation.kind=='linear' and p.get('type')==2 and p.get('r')==4096:
                            key={(2048,6144):'conv-in',(2048,2048):'projection',(11776,2048):'dense-down'}.get((p['k'],p['o']))
                            if key is not None:module=modules[key];counts['projection']+=1
                        if variant in ('square64','ordered64','projection-ordered64') and operation.kind=='attention':
                            module=modules['ordered64' if variant=='projection-ordered64' else variant];counts['attention']+=1
                        if module is None:plan.append(old)
                        else:
                            b=call_bindings(old[0],old[1]);plan.bind(module,tuple(b[a['name']] for a in module.manifest['arguments']),label=old[2])
                    graph=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+tuple(modules.values()))
                    owned.callback(graph.close);candidates[variant][phase]=graph
                    print('OVERLAY',variant,phase,counts,flush=True)
            def select(variant):
                for phase,forward in forwards.items():forward.graph=candidates.get(variant,original)[phase]
            owned.callback(lambda:select('original'))
            teacher={}
            saved={}
            for variant in ('original-before','ordered64','projection-ordered64','original-after'):
                select(variant)
                print('MEASURE',variant,flush=True)
                with Session(model,batch=1,chunk=4096,timeout=120) as session:
                    measured=run(session,Case(1,8192,128),warmup=1,repeats=3)
                    session.reset();ids=list(session.prefill(prompts(model.config.vocab,1,8192)))
                    for _ in range(127):ids.extend(session.decode(1))
                    same=ids==expected
                    mismatch=next((i for i,(a,b) in enumerate(zip(ids,expected)) if a!=b),None)
                    session.reset();session.prefill(prompts(model.config.vocab,1,8192))
                    logits=[]
                    for step in range(9):
                        values=session.read_feature('logits').reshape(-1,model.config.vocab)[-1].copy()
                        assert np.isfinite(values).all()
                        logits.append(values)
                        if step<8:session.step([expected[step]])
                teacher[variant]=np.asarray(logits)
                saved[variant]=teacher[variant]
                left=teacher['original-before'].astype('float64');right=teacher[variant].astype('float64')
                lp=left-left.max(-1,keepdims=True);lp-=np.log(np.exp(lp).sum(-1,keepdims=True))
                lq=right-right.max(-1,keepdims=True);lq-=np.log(np.exp(lq).sum(-1,keepdims=True))
                top=np.argsort(left,axis=-1)[:,-2:]
                check=dict(variant=variant,native=measured,free_ar_identical=same,first_mismatch=mismatch,output_ids=ids,
                    teacher_top1_agreement=float(np.mean(left.argmax(-1)==right.argmax(-1))),relative_logit_error=float(np.linalg.norm(left-right)/max(np.linalg.norm(left),1e-30)),max_absolute_logit_error=float(np.max(np.abs(left-right))),mean_kl=float(np.mean(np.sum(np.exp(lp)*(lp-lq),axis=-1))),
                    original_top1_margin=[float(left[s,t[1]]-left[s,t[0]]) for s,t in enumerate(top)],
                    eligible_for_promotion=same)
                meta.setdefault('comparisons',[]).append(check)
                write(root/'report.json',report)
                np.savez_compressed(root/'teacher-logits.npz',**saved)
                print('RESULT',variant,'pp',measured['summary']['prefill_tokens_per_second']['median'],'tg',measured['summary']['decode_tokens_per_second']['median'],'same',same,'mismatch',mismatch,'rel',check['relative_logit_error'],'max',check['max_absolute_logit_error'],flush=True)
                if variant.startswith('original') and not same:raise ValueError('Current original no longer matches the retained FP8 output')
    passed=True
except BaseException:
    meta['traceback']=traceback.format_exc();print(meta['traceback'],flush=True)
finally:
    rec.program(program)
    rec.artifact(__file__,'diagnostic/runner.py')
    for name,path in paths.items():rec.artifact(path,'overlay/'+name+'.tbin')
    rec.artifact('build/lfm-moe/attention_ordered_rows.py','kernels/attention_ordered_rows.py')
    rec.artifact('build/lfm-moe/compile-attention-ordered-rows.py','kernels/compile-attention-ordered-rows.py')
    if (root/'teacher-logits.npz').exists():rec.artifact(root/'teacher-logits.npz','diagnostic/teacher-logits.npz')
    write(root/'report.json',report);rec.update(report,final=True,status='passed' if passed else 'failed');print('RETAINED',rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

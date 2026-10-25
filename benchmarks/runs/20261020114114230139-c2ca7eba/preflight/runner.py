"""Bounded manual algorithm A/B checks with standardized per-configuration records."""
import json,time,resource,traceback
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from akas.models.lfm2.model import LFM2MoE
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.tensor.executor import bind
from akas.optimization.evaluation import diagnostic,measured_requests
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder,write
from akas.engine import Engine
from akas.config import EngineConfig

root=Path('build/lfm-moe/prototype-model-v2');root.mkdir(exist_ok=True)
old=Path('build/single-user-long-context/lfm2.5-24b-a2b-q4_0-cuda-default-ar/jobs/ee9b3acb399892d0/attempt-0001')
oldreport=json.loads((old/'report.json').read_text());expected=np.load(old/'diagnostics.npz',allow_pickle=False)
checkpoint='build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
records=[]

def attach(executor,device,paths):
    modules={};replacements=[]
    try:
        for key,path in paths.items():modules[key]=device.load(path)
        for phase,chunk in (('decode',1),('prefill',512),('prefill_state',512)):
            pf=executor.batches.select(1,chunk,phase=phase).forward
            plan=ExecutionPlan(device,labeled=True);changed=0
            for entry,call in zip(pf.plan,pf.definition.calls):
                key=(phase,call.kind,call.parameters.get('k'))
                if key not in modules:key=(phase,call.kind,None)
                if key in modules:
                    module=modules[key];bindings=call_bindings(entry[0],entry[1])
                    args=tuple(bindings[a['name']] for a in module.manifest['arguments'])
                    plan.append((module,bind(device,module,args),entry[2]));changed+=1
                else:plan.append(entry)
            if changed:
                graph=plan.capture(resources=tuple(pf.buffers.values())+tuple(pf.kernels.values())+tuple(modules.values()))
                replacements.append((pf,pf.graph,graph));pf.graph=graph
        return modules,replacements
    except BaseException:
        detach(modules,replacements);raise

def detach(modules,replacements):
    for pf,previous,graph in replacements:pf.graph=previous;graph.close()
    for m in modules.values():m.release()

def evaluate(name,paths,prefixes=(8192,)):
    metadata=dict(runtime='akas',purpose='preflight',protocol='native-token-step-v2',
                  model='akas.models.lfm2.model:LFM2MoE',provider='cuda',target='sm_89',
                  checkpoint=checkpoint,checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
                  weight_format='q4_0',kv_dtype='float16',mode='ar',graphs=True,chunk=512,
                  device=dict(name='NVIDIA L4',provider='cuda'),warmup=1,repeats=3,
                  shape_timeout_seconds=120,manual_variant=name,
                  qualification='In-process prototype substitution; independent operation checks follow before promotion; diagnostics and 128 greedy IDs versus retained full-model reference; no HTTP',
                  prototype_paths={str(k):str(v) for k,v in paths.items()},baseline_run_id='20261020102328416670-be712fbb')
    rec=Recorder('benchmarks',metadata,kind='algorithm',change='Manually optimize measured expert and attention bottlenecks: '+name,parent=metadata['baseline_run_id'])
    report=dict(schema='akas.benchmark.v1',metadata=metadata,results=[]);passed=False
    modules={};replacements=[]
    try:
        modules,replacements=attach(executor,device,paths)
        for prefix in prefixes:
            print('MEASURE',name,prefix,flush=True)
            values=diagnostic(executor,dict(chunk=512),1,prefix,timeout=120)
            reference=expected[f'b1-p{prefix}']
            relative=float(np.linalg.norm(values.astype('float64')-reference)/max(np.linalg.norm(reference),1e-30))
            if relative>.01:raise ValueError(f'logit drift {relative} at {prefix}')
            with Session(executor,batch=1,chunk=512,timeout=120) as session:
                row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
            measured=measured_requests(engine,1,prefix,128,timeout=120)
            oldrow=next(x for x in oldreport['results'] if x['case']['prompt']==prefix)
            same=measured['output_ids']==oldrow['output_ids']
            row.update(correctness=dict(correct=same,reference='20261020102328416670-be712fbb',relative_logit_error=relative),output_ids=measured.pop('output_ids'),engine=measured)
            report['results'].append(row)
            write(root/(name+'.json'),report)
            print('RESULT',name,prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'relative',relative,'greedy',same,flush=True)
            if not same:raise ValueError('greedy output differs')
        passed=True
    except Exception as e:
        metadata['error']=f'{type(e).__name__}: {e}';metadata['traceback']=traceback.format_exc();print(metadata['traceback'],flush=True)
    finally:
        detach(modules,replacements)
        metadata['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        rec.artifact(__file__,'preflight/runner.py')
        for f in ('packed_expert.py','packed_prefill.py','q4_tile.py','mma_decode.py'):
            rec.artifact(Path('build/lfm-moe')/f,'preflight/'+f)
        for index,path in enumerate(dict.fromkeys(paths.values())):
            manifest,payloads=read_artifact(path)
            write(root/'manifest.json',manifest);rec.artifact(root/'manifest.json',f'kernels/{index}/manifest.json')
            # Retain portable compiler input, not a copy of the GPU binary.
            for key,payload in payloads.items():
                if key.endswith(('.json','.wgsl','.cu','.py')) or key in ('tir','ir'):
                    local=root/('payload-'+str(index));local.write_bytes(payload)
                    rec.artifact(local,f'kernels/{index}/'+key.replace('/','_'))
        rec.update(report,final=True,status='passed' if passed else 'failed')
        print('RETAINED',name,rec.id,passed,flush=True)
    records.append(dict(name=name,passed=passed,run_id=rec.id,report=report))
    return passed

packed={('decode','quantized_experts',k):Path(f'build/lfm-moe/packed-experts/{k}-128.tbin') for k in (2048,1536)}
with tensor.Device(provider='cuda') as device:
    with LFM2MoE(checkpoint,'build/lfm-moe/manual-v2/cuda/program',device,graphs=True) as executor:
        engine=Engine(executor,config=EngineConfig(prefill_chunk=512,prefill_requests=1))
        evaluate('baseline-before',{})
        prefill_variants=[];attention_variants=[]
        for bm,bn,bk in ((16,64,32),(32,64,128),(64,128,128)):
            paths={**packed,**{(phase,'quantized_experts',k):Path(f'build/lfm-moe/half2-prefill/{k}-{bm}-{bn}-{bk}.tbin') for phase in ('prefill','prefill_state') for k in (2048,1536)}}
            name=f'half2-{bm}-{bn}-{bk}'
            if evaluate(name,paths):prefill_variants.append((records[-1],paths))
        for bk in (64,128):
            paths={**packed,('decode','attention_partial',None):Path(f'build/lfm-moe/mma-decode/{bk}.tbin')}
            if evaluate(f'mma-decode-{bk}',paths,prefixes=(32,8192)):attention_variants.append((records[-1],paths))
        if prefill_variants and attention_variants:
            bestpp=max(prefill_variants,key=lambda x:x[0]['report']['results'][0]['summary']['prefill_tokens_per_second']['median'])
            besttg=max(attention_variants,key=lambda x:x[0]['report']['results'][-1]['summary']['decode_tokens_per_second']['median'])
            combined={**bestpp[1],**besttg[1]}
            evaluate('combined',combined,prefixes=(32,8192,130944))
        evaluate('baseline-after',{})
expected.close()
print('FINISHED',json.dumps([dict(name=r['name'],passed=r['passed'],run_id=r['run_id']) for r in records]),flush=True)

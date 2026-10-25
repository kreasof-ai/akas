"""In-process expert substitution; no extra model upload or tuning campaign."""
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.tensor.executor import bind
from akas.optimization.evaluation import diagnostic,measured_requests
from akas.benchmarking.runner import run
from akas.benchmarking.cases import Case
from akas.engine import Engine
from akas.config import EngineConfig

old=Path('build/single-user-long-context/lfm2.5-24b-a2b-q4_0-cuda-default-ar/jobs/ee9b3acb399892d0/attempt-0001')
expected=np.load(old/'diagnostics.npz',allow_pickle=False)
oldreport=json.loads((old/'report.json').read_text())
frame=executor.batch(1);original=frame.forward.graph
engine=Engine(executor,config=EngineConfig(prefill_chunk=512,prefill_requests=1))
experiments=[]
for threads in (128,256):
    modules={}
    graph=None
    try:
        for depth in (2048,1536):
            path=Path(f'build/lfm-moe/packed-experts/{depth}-{threads}.tbin')
            # Another process only compiles/validates the four bounded variants.
            deadline=time.monotonic()+180
            while not path.exists() and time.monotonic()<deadline:
                time.sleep(0.25)
            modules[depth]=device.load(path)
        plan=ExecutionPlan(device,labeled=True)
        for entry,call in zip(frame.forward.plan,frame.forward.definition.calls):
            if call.kind=='quantized_experts':
                module=modules[call.parameters['k']]
                bindings=call_bindings(entry[0],entry[1])
                args=tuple(bindings[a['name']] for a in module.manifest['arguments'])
                plan.append((module,bind(device,module,args),entry[2]))
            else:plan.append(entry)
        graph=plan.capture(resources=tuple(frame.forward.buffers.values())+tuple(frame.forward.kernels.values())+tuple(modules.values()))
        frame.forward.graph=graph
        experiment=dict(threads=threads,results=[])
        for prefix in (32,8192):
            values=diagnostic(executor,dict(chunk=512),1,prefix,timeout=120)
            ref=expected[f'b1-p{prefix}']
            relative=float(np.linalg.norm(values.astype('float64')-ref)/max(np.linalg.norm(ref),1e-30))
            with Session(executor,batch=1,chunk=512,timeout=120) as session:
                row=run(session,Case(1,prefix,128),warmup=1,repeats=3)
            measured=measured_requests(engine,1,prefix,128,timeout=120)
            previous=next(x for x in oldreport['results'] if x['case']['prompt']==prefix)
            greedy=measured['output_ids']==previous['output_ids']
            experiment['results'].append(dict(prefix=prefix,relative_logit_error=relative,greedy_identical=greedy,native=row,engine=measured))
            print('PACKED',threads,prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'relative',relative,'greedy',greedy,flush=True)
        experiments.append(experiment)
    finally:
        frame.forward.graph=original
        if graph is not None:graph.close()
        for m in modules.values():m.release()
metadata['packed_experiments']=experiments
write(root/'packed.json',experiments)
rec.artifact('build/lfm-moe/packed_expert.py','kernels/packed_expert.py')
rec.artifact(__file__ if False else 'build/lfm-moe/profile-followup.py','kernels/followup.py')
rec.artifact(root/'packed.json','kernels/packed.json')
prefill_experiments=[]
for bm,bn,bk in ((32,64,128),(32,128,128),(64,128,128),(16,128,128),(32,64,64)):
    modules={}; replacements=[]
    try:
        for depth in (2048,1536):
            path=Path(f'build/lfm-moe/prefill-experts/{depth}-{bm}-{bn}-{bk}.tbin')
            deadline=time.monotonic()+180
            while (not path.exists()) and time.monotonic()<deadline:time.sleep(.25)
            modules[depth]=device.load(path)
        for phase in ('prefill','prefill_state'):
            pf=executor.batches.select(1,512,phase=phase).forward
            plan=ExecutionPlan(device,labeled=True)
            for entry,call in zip(pf.plan,pf.definition.calls):
                if call.kind=='quantized_experts':
                    module=modules[call.parameters['k']]
                    bindings=call_bindings(entry[0],entry[1])
                    args=tuple(bindings[a['name']] for a in module.manifest['arguments'])
                    plan.append((module,bind(device,module,args),entry[2]))
                else:plan.append(entry)
            graph=plan.capture(resources=tuple(pf.buffers.values())+tuple(pf.kernels.values())+tuple(modules.values()))
            replacements.append((pf,pf.graph,graph));pf.graph=graph
        values=diagnostic(executor,dict(chunk=512),1,8192,timeout=120)
        ref=expected['b1-p8192']
        relative=float(np.linalg.norm(values.astype('float64')-ref)/max(np.linalg.norm(ref),1e-30))
        with Session(executor,batch=1,chunk=512,timeout=120) as session:
            row=run(session,Case(1,8192,128),warmup=1,repeats=2)
        measured=measured_requests(engine,1,8192,128,timeout=120)
        previous=next(x for x in oldreport['results'] if x['case']['prompt']==8192)
        greedy=measured['output_ids']==previous['output_ids']
        prefill_experiments.append(dict(block_m=bm,columns=bn,block_k=bk,relative_logit_error=relative,greedy_identical=greedy,native=row,engine=measured))
        print('PREFILL_EXPERT',bm,bn,bk,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'relative',relative,'greedy',greedy,flush=True)
    finally:
        for pf,previous,graph in replacements:pf.graph=previous;graph.close()
        for m in modules.values():m.release()
metadata['prefill_experiments']=prefill_experiments
write(root/'prefill-experiments.json',prefill_experiments)
rec.artifact(root/'prefill-experiments.json','kernels/prefill-experiments.json')
expected.close()
# Outer owner closes the executor; do not close the Engine here.

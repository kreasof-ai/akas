"""Run the normal AKAS compiler search on retained H200 phase operations."""
from pathlib import Path
import modal
from tools.h200_modal import ROOT, volume
from tools.h200_qualify import compiler_image

app=modal.App('akas-h200-builtin-search')


def operation_bundle(program, phase, key):
    """Preserve the phase's exact bindings and incumbent schedule."""
    import json
    from akas.graph import Graph
    from akas.compilation.compile import plan
    from akas.compilation.liveness import prune_buffers
    from tensor.compiler.search import ScheduleProfile

    forward=json.loads((program/phase['path']).read_text())
    record=forward['kernels'][key]
    graph=Graph.from_metadata(forward['forward'])
    graph.calls=[c for c in graph.calls if c.key==key]
    graph.request=None
    prune_buffers(graph)
    schedules=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_90a',
        entries=[dict(operation=record['kind'],parameters=record['parameters'],schedule=record['schedule'])]))
    return plan(graph,provider='cuda',target='sm_90a',schedules=schedules),record


@app.function(image=compiler_image,gpu='H200',volumes={'/data':volume},timeout=1200,
              cpu=8,memory=65536,scaledown_window=2,max_containers=1)
def search(program: str, phase_name: str, batch: int, budget: int):
    import hashlib,json,os,time,traceback
    import tensor
    from akas.compilation.compile import build
    from akas.compilation.tuning import SearchSettings
    from akas.benchmarking.history import Recorder

    source=Path('/data/programs')/program
    manifest=json.loads((source/'program.json').read_text())
    phase=next(p for p in manifest['phases'] if p['phase']==phase_name and p['rows']==batch)
    forward=json.loads((source/phase['path']).read_text())
    source_sha=hashlib.sha256((source/'program.json').read_bytes()).hexdigest()
    work=Path('/data/builtin-search')/program/(phase_name+'-b'+str(batch))
    work.mkdir(parents=True,exist_ok=True)
    report_path=work/'search.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(
        schema='akas.builtin-search-evidence.v1',program=program,program_sha256=source_sha,
        phase=phase_name,batch=batch,cases=[])
    if report['program_sha256']!=source_sha:raise ValueError('search program changed')
    metadata=dict(runtime='akas',purpose='builtin-qualified-search',model='LiquidAI/LFM2.5-24B-A2B-FP8',
        provider='cuda',target='sm_90a',weight_format='fp8',mode='ar',kv_dtype='float16',
        program=program,program_sha256=source_sha,phase=phase_name,batch=batch,
        protocol='akas-builtin-qualified-search-v1',search_budget_per_hot_operation=budget,
        independent_reference_recipes=True)
    recorder=Recorder('/data/standard-benchmarks',metadata,kind='search',
        change='Run built-in compiler qualification and beam search from the retained H200 incumbent',
        dependency_sources={'tensor':Path('/workspace/tensor')})
    recorder.artifact(Path(__file__),'benchmark.py')
    log=[]
    def emit(*items):
        line=' '.join(map(str,items));log.append(line);print(line,flush=True)
    priority={'fp8_expert_swiglu':0,'expert_pair_partial':0,'expert_swiglu':0,
              'expert_partial':1,'linear':2,'projection_partial':3,'attention':4,'attention_partial':4}
    keys=sorted(forward['kernels'],key=lambda k:(priority.get(forward['kernels'][k]['kind'],10),k))
    completed={c['key'] for c in report['cases'] if c['status']=='passed'}
    tick=time.monotonic();status='failed'
    emit('STARTED',recorder.id,program,phase_name,batch,'operations',len(keys))
    try:
        with tensor.Device() as device:
            metadata['device']=device.info
            for key in keys:
                if key in completed:continue
                if time.monotonic()-tick>1020:
                    emit('CHECKPOINT remaining operations for resume');break
                original=forward['kernels'][key]
                kind=original['kind'];hot=kind in priority
                trials=budget if hot else 1
                os.environ['AKAS_SKIP_AUTOTUNE']='0' if hot else '1'
                task=dict(key=key,kind=kind,parameters=original['parameters'],incumbent=original['schedule'],
                          source_artifact_sha256=original['sha256'],budget=trials)
                emit('BUILTIN SEARCH',key,kind,'budget',trials)
                try:
                    bundle,_=operation_bundle(source,phase,key)
                    produced=build(bundle,work/key,search=True,device=device,
                        search_settings=SearchSettings(budget=trials,width=4,warmup=1,samples=3,repeats=3,
                            cache_working_set_bytes=128*1024**2 if hot else 0))
                    task.update(status='passed',kernel=produced['kernels'][key],
                                qualification=produced['tuning'],artifact_root=str(work/key))
                    selected=produced['tuning']['operations'][0]['selected']
                    emit('QUALIFIED',key,kind,'trials',selected['trials'],'score',selected['score'])
                    completed.add(key)
                except Exception as error:
                    task.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
                    emit('QUALIFICATION FAILED',key,kind,task['error'][:500])
                report['cases']=[c for c in report['cases'] if c['key']!=key]+[task]
                report_path.write_text(json.dumps(report,indent=2));volume.commit()
        status='passed' if len(completed)==len(keys) else 'failed' if any(c['status']=='failed' for c in report['cases']) else 'partial'
    finally:
        report.update(status=status,elapsed_seconds=time.monotonic()-tick)
        report_path.write_text(json.dumps(report,indent=2))
        metadata.update(qualification_status=status,completed_operations=len(completed),total_operations=len(keys))
        recorder.artifact(report_path,'builtin-search.json')
        worker=recorder.directory/'worker.log';worker.write_text('\n'.join(log)+'\n');recorder.artifact(worker,'worker.log')
        for task in report['cases']:
            p=work/task['key']/'forward.json'
            if p.exists():recorder.artifact(p,'qualified-operations/'+task['key']+'.json')
        record=recorder.update(dict(metadata=metadata,results=[]),final=True,status=status)
        volume.commit()
    emit('SEARCH RUN',recorder.id,record['status'],len(completed),'/',len(keys))
    return dict(run_id=recorder.id,status=record['status'],out=str(work),completed=len(completed),total=len(keys))


@app.local_entrypoint()
def main(program: str='optimized-v5',phase: str='decode',batch: int=1,budget: int=16):
    import json,time
    ledger_path=ROOT/'build/h200/cost-ledger.json';ledger=json.loads(ledger_path.read_text())
    rate=.001261+8*.0000131+64*.00000222
    if sum(r['conservative_usd'] for r in ledger)+1200*rate>28:raise RuntimeError('insufficient campaign budget')
    row=dict(action='akas-builtin-search',program=program,phase=phase,batch=batch,budget=budget,
        status='reserved',started_at=time.time(),conservative_usd=1200*rate)
    ledger.append(row);ledger_path.write_text(json.dumps(ledger,indent=2));tick=time.monotonic()
    try:
        result=search.remote(program,phase,batch,budget);row.update(status='completed',result=result);print(json.dumps(result),flush=True)
    finally:
        row['elapsed_seconds']=time.monotonic()-tick
        if row['status']=='completed':row['conservative_usd']=(row['elapsed_seconds']+30)*rate
        ledger_path.write_text(json.dumps(ledger,indent=2))

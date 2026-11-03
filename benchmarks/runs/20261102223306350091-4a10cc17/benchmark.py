"""Run the normal AKAS compiler search on retained H200 phase operations."""
from pathlib import Path
import modal
from tools.h200_modal import ROOT, volume
from tools.h200_qualify import compiler_image

app=modal.App('akas-h200-builtin-search')


def prime_offline_cache():
    """Reuse L40S-produced artifacts through the normal checked source cache."""
    pool=Path('/data/builtin-compiler-cache');pool.mkdir(exist_ok=True)
    print('OFFLINE COMPILER CACHE',len(list(pool.glob('*.tbin'))),'precompiled artifacts',flush=True)
    return pool


def install_cached_operation(operation_out, original, shared=None):
    """Copy matching cached sources inside the validated bundle boundary."""
    import json,shutil
    shared=Path('/data/builtin-compiler-cache') if shared is None else Path(shared)
    index=shared/'index.json'
    operation_out.mkdir(exist_ok=True)
    cache=operation_out/'search-kernels'
    if cache.is_symlink():cache.unlink()
    cache.mkdir(exist_ok=True)
    if not index.exists():return
    for entry in json.loads(index.read_text()):
        if entry['kind']!=original['kind'] or entry['parameters']!=original['parameters']:continue
        for digest in entry['digests']:
            for suffix in ('.py','.tbin'):
                source=shared/(digest+suffix);destination=cache/source.name
                if source.exists() and not destination.exists():shutil.copyfile(source,destination)


def operation_bundle(program, phase, key):
    """Preserve the phase's exact bindings and incumbent schedule."""
    import json
    from akas.graph import Graph
    from akas.compilation.compile import plan
    from akas.compilation.liveness import prune_buffers
    from tensor.compiler.search import ScheduleProfile
    from dataclasses import replace
    from akas.kernels.registry import candidates

    forward=json.loads((program/phase['path']).read_text())
    record=forward['kernels'][key]
    graph=Graph.from_metadata(forward['forward'])
    graph.calls=[c for c in graph.calls if c.key==key]
    graph.request=None
    prune_buffers(graph)
    schedules=ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_90a',
        entries=[dict(operation=record['kind'],parameters=record['parameters'],schedule=record['schedule'])]))
    variants=candidates(record['kind'],record['parameters'],contract=graph.contract,
        provider='cuda',phase=graph.phase,target='sm_90a')
    # A conditional expert TMA schedule hung on H200; retain the failure and
    # exclude that execution path until its compiler support is repaired.
    # This changes search legality, never kernel arithmetic or tolerances.
    catalog=tuple(replace(item,space=tuple((name,(False,) if name=='tma' else values)
                                          for name,values in item.space))
                  for item in variants)
    return plan(graph,provider='cuda',target='sm_90a',schedules=schedules,catalog=catalog),record


@app.function(image=compiler_image,gpu='H200',volumes={'/data':volume},timeout=1200,
              cpu=8,memory=65536,scaledown_window=2,max_containers=1)
def search(program: str, phase_name: str, batch: int, budget: int, seconds: int=1020):
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
        independent_reference_recipes=True,
        excluded_schedule_axes=dict(tma=True),
        excluded_schedule_reason='H200 conditional expert TMA execution hang; /data/results/builtin-tma-hang-evidence.zip')
    recorder=Recorder('/data/standard-benchmarks',metadata,kind='search',
        change='Run built-in compiler qualification and beam search from the retained H200 incumbent',
        dependency_sources={'tensor':Path('/workspace/tensor')})
    recorder.artifact(Path(__file__),'benchmark.py')
    hang=Path('/data/results/builtin-tma-hang-evidence.zip')
    if hang.exists():recorder.artifact(hang,'rejected-tma-candidate.zip')
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
                if time.monotonic()-tick>seconds:
                    emit('CHECKPOINT remaining operations for resume');break
                original=forward['kernels'][key]
                kind=original['kind'];hot=kind in priority
                trials=(budget if kind.startswith('expert') or kind=='fp8_expert_swiglu'
                        else min(budget,4)) if hot else 1
                os.environ['AKAS_SKIP_AUTOTUNE']='0' if hot else '1'
                task=dict(key=key,kind=kind,parameters=original['parameters'],incumbent=original['schedule'],
                          source_artifact_sha256=original['sha256'],budget=trials)
                emit('BUILTIN SEARCH',key,kind,'budget',trials)
                try:
                    install_cached_operation(work/key,original)
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
            if task['status']=='passed':
                artifact=Path(task['artifact_root'])/task['kernel']['path']
                recorder.artifact(artifact,'qualified-kernels/'+task['key']+'.tbin')
                generated=artifact.with_suffix('.py')
                if generated.exists():recorder.artifact(generated,'qualified-kernels/'+task['key']+'.py')
        record=recorder.update(dict(metadata=metadata,results=[]),final=True,status=status)
        volume.commit()
    emit('SEARCH RUN',recorder.id,record['status'],len(completed),'/',len(keys))
    return dict(run_id=recorder.id,status=record['status'],out=str(work),completed=len(completed),total=len(keys))


@app.function(image=compiler_image,gpu='H200',volumes={'/data':volume},timeout=3600,
              cpu=8,memory=65536,scaledown_window=2,max_containers=1)
def campaign(program: str, budget: int=16):
    """Qualify every used geometry and search the main projections first."""
    import json,time
    from tools.h200_apply_builtin import assemble
    prime_offline_cache()
    source=Path('/data/programs')/program
    manifest=json.loads((source/'program.json').read_text())
    # Group size 16 executes prefill geometries 1, 4 and 16 for this workload.
    priorities=[('decode',1),('decode',256),('prefill',16),('decode',4),('decode',16),
                ('decode',64),('prefill',4),('prefill',1)]
    priorities += [('prefill_state',b) for b in (1,4,16)]
    tick=time.monotonic();results=[]
    for phase,batch in priorities:
        remaining=3400-(time.monotonic()-tick)
        if remaining<90:break
        results.append(search.local(program,phase,batch,4 if (phase,batch)==('decode',1) else budget,
                                    seconds=min(900,int(remaining)-60)))
    reports=sorted((Path('/data/builtin-search')/program).glob('*/search.json'))
    # Keep unused compiled geometries; coverage remains explicit per phase.
    destination=Path('/data/programs')/(program+'-builtin-'+time.strftime('%H%M%S'))
    assembled=assemble(source,destination,reports)
    volume.commit()
    return dict(results=results,assembled=assembled,elapsed_seconds=time.monotonic()-tick)


@app.local_entrypoint()
def main(program: str='optimized-v5',phase: str='decode',batch: int=1,budget: int=16,
         all_phases: bool=False):
    import json,time
    ledger_path=ROOT/'build/h200/cost-ledger.json';ledger=json.loads(ledger_path.read_text())
    rate=.001261+8*.0000131+64*.00000222
    bound=3600 if all_phases else 1200
    if sum(r['conservative_usd'] for r in ledger)+bound*rate>28:raise RuntimeError('insufficient campaign budget')
    row=dict(action='akas-builtin-search',program=program,phase=phase,batch=batch,budget=budget,
        status='reserved',started_at=time.time(),conservative_usd=bound*rate,all_phases=all_phases)
    ledger.append(row);ledger_path.write_text(json.dumps(ledger,indent=2));tick=time.monotonic()
    try:
        result=campaign.remote(program,budget) if all_phases else search.remote(program,phase,batch,budget)
        row.update(status='completed',result=result);print(json.dumps(result),flush=True)
    finally:
        row['elapsed_seconds']=time.monotonic()-tick
        if row['status']=='completed':row['conservative_usd']=(row['elapsed_seconds']+30)*rate
        ledger_path.write_text(json.dumps(ledger,indent=2))

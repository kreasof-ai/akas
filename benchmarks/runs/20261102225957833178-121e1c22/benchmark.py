"""Stop tuning, apply qualified winners and numerical repairs, measure once."""
from pathlib import Path
import modal
from tools.h200_modal import ROOT,volume
from tools.h200_qualify import compiler_image

app=modal.App('akas-h200-quick-measure')


@app.function(image=compiler_image,gpu='H200',volumes={'/data':volume},timeout=600,
              cpu=8,memory=65536,scaledown_window=2,max_containers=1)
def measure():
    import hashlib,json,os,shutil,time
    import tensor
    from tensor.artifacts.format import read_artifact
    from akas.artifacts.read import artifact_path,read_program
    from akas.compilation.compile import build
    from akas.compilation.tuning import SearchSettings
    from akas.benchmarking.history import Recorder
    from tools.h200_builtin_search import operation_bundle
    from tools.h200_apply_builtin import assemble
    from tools.h200_standard import standard_akas

    source=Path('/data/programs/optimized-v5')
    destination=Path('/data/programs')/('builtin-measured-'+time.strftime('%H%M%S'))
    reports=sorted(Path('/data/builtin-search/optimized-v5').glob('*/search.json'))
    print('ASSEMBLE',assemble(source,destination,reports),flush=True)
    manifest=json.loads((destination/'program.json').read_text())
    source_manifest=json.loads((source/'program.json').read_text())
    metadata=dict(runtime='akas',purpose='numerical-repair-qualification',
        model='LiquidAI/LFM2.5-24B-A2B-FP8',provider='cuda',target='sm_90a',
        program=destination.name,protocol='builtin-independent-operation-references-v1',
        scope='Attention state addressing and compensated rotary phase; existing tolerances unchanged')
    recorder=Recorder('/data/standard-benchmarks',metadata,kind='qualification',
        change='Qualify numerical repairs before the requested immediate benchmark',
        dependency_sources={'tensor':Path('/workspace/tensor')})
    recorder.artifact(Path(__file__),'benchmark.py');cases=[]
    os.environ['AKAS_SKIP_AUTOTUNE']='1'
    with tensor.Device() as device:
        metadata['device']=device.info
        for phase in manifest['phases']:
            if phase['phase'] not in ('decode','prefill'):continue
            if phase['phase']=='prefill' and phase['rows'] not in (1,4,16):continue
            original_phase=next(p for p in source_manifest['phases']
                if (p['phase'],p['rows'],p['chunk'])==(phase['phase'],phase['rows'],phase['chunk']))
            path=destination/phase['path'];forward=json.loads(path.read_text())
            for key,original in list(forward['kernels'].items()):
                if original['kind'] not in ('rope','attention','attention_partial'):continue
                print('QUALIFY REPAIR',phase['phase'],phase['rows'],original['kind'],key,flush=True)
                bundle,_=operation_bundle(source,original_phase,key)
                work=Path('/data/quick-repairs')/key
                produced=build(bundle,work,search=True,device=device,
                    search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1))
                record=dict(produced['kernels'][key]);candidate=artifact_path(work,record)
                old,_=read_artifact(artifact_path(destination,original));new,_=read_artifact(candidate)
                abi=lambda m:[(v['name'],v['dtype'],v['shape']) for v in m['arguments']]
                if abi(old)!=abi(new):raise ValueError('repair changed ABI')
                target=destination/'qualified-kernels'/(record['sha256']+'.tbin')
                target.parent.mkdir(exist_ok=True);shutil.copyfile(candidate,target)
                shutil.copyfile(candidate.with_suffix('.py'),target.with_suffix('.py'))
                record['path']=target.relative_to(destination).as_posix();forward['kernels'][key]=record
                cases.append(dict(phase=phase['phase'],rows=phase['rows'],key=key,
                                  kernel=record,qualification=produced['tuning']))
                recorder.artifact(candidate,'kernels/'+key+'.tbin')
                recorder.artifact(candidate.with_suffix('.py'),'kernels/'+key+'.py')
            path.write_text(json.dumps(forward,indent=2));phase['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (destination/'program.json').write_text(json.dumps(manifest,indent=2));read_program(destination)
    metadata['program_sha256']=hashlib.sha256((destination/'program.json').read_bytes()).hexdigest()
    metadata['repaired_operations']=len(cases)
    raw=recorder.directory/'repairs.json';raw.write_text(json.dumps(cases,indent=2));recorder.artifact(raw,'repairs.json')
    recorder.update(dict(metadata=metadata,results=[]),final=True,status='passed');volume.commit()
    os.environ['AKAS_SKIP_AUTOTUNE']='0'
    print('REPAIRS PASSED',recorder.id,len(cases),destination.name,flush=True)
    result=standard_akas.local(program_name=destination.name,prefill_group_size=16)
    result.update(program=destination.name,repair_run=recorder.id)
    volume.commit();return result


@app.local_entrypoint()
def main():
    import json,time
    path=ROOT/'build/h200/cost-ledger.json';ledger=json.loads(path.read_text())
    rate=.001261+8*.0000131+64*.00000222
    if sum(r['conservative_usd'] for r in ledger)+600*rate>28:raise RuntimeError('insufficient budget')
    row=dict(action='akas-immediate-benchmark',status='reserved',started_at=time.time(),conservative_usd=600*rate)
    ledger.append(row);path.write_text(json.dumps(ledger,indent=2));tick=time.monotonic()
    try:
        result=measure.remote();row.update(status='completed',result=result);print(json.dumps(result),flush=True)
    finally:
        row['elapsed_seconds']=time.monotonic()-tick
        if row['status']=='completed':row['conservative_usd']=(row['elapsed_seconds']+30)*rate
        path.write_text(json.dumps(ledger,indent=2))

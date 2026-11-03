"""Budgeted built-in search and retained-reference qualification on H200."""
from pathlib import Path
import modal
from tools.h200_modal import ROOT, aimage, volume

app = modal.App('akas-h200-qualified-search')
compiler_image = (aimage.add_local_dir(ROOT/'build/nvrtc-12.9', '/opt/tensor-nvrtc-12.9', copy=True)
                  .env({'TENSOR_NVRTC_HOME':'/opt/tensor-nvrtc-12.9',
                        'OPENBLAS_NUM_THREADS':'8', 'OMP_NUM_THREADS':'8'}))


@app.function(image=aimage, gpu='H200', volumes={'/data':volume}, timeout=600,
              cpu=4, memory=65536, scaledown_window=2, max_containers=1)
def numerical(program: str):
    import hashlib
    import json
    import tempfile
    import zipfile
    import numpy as np
    import tensor
    from akas.models.lfm2 import LFM2MoE
    from akas.benchmarking.session import Session
    from akas.benchmarking.history import Recorder

    bundle=Path('/data/programs')/program
    temporary=tempfile.TemporaryDirectory()
    refs=Path(temporary.name)
    with zipfile.ZipFile('/data/qualification/lfm2.5-24b-references.zip') as z:
        z.extractall(refs)
    provenance=json.loads((refs/'provenance.json').read_text())
    for item in provenance['source_files']:
        path=refs/item.get('retained_as',item['file'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('retained reference checksum mismatch')
    old=json.loads((refs/'akas_results.json').read_text())
    results={r:json.loads((refs/(r+'_results.json')).read_text()) for r in ['vllm','sglang']}
    arrays={r:np.load(refs/(r+'_logits.npz')) for r in ['akas','vllm','sglang']}
    fp64=np.load(refs/'fp64-reference.npz')
    metadata=dict(runtime='akas',purpose='numerical-qualification',model='LiquidAI/LFM2.5-24B-A2B-FP8',
        model_revision='f54bf39547fd5b7955abb9fe8d1b151e31fbbddb',provider='cuda',target='sm_90a',
        weight_format='fp8',kv_dtype='float16',program=program,mode='ar',
        program_sha256=hashlib.sha256((bundle/'program.json').read_bytes()).hexdigest(),
        qualified_akas_run_id='20261102212133108734-4a2dd18e' if program=='optimized-v5' else None,
        reference_qualification_run_id=provenance['qualification_run_id'],
        qualification_status='running',protocol='retained-l40s-numerical-transfer-v1',
        acceptance=dict(bos_three_way_first_tokens=8,chat_three_way_first_tokens=8,
                        finite_logits=True,max_normalized_logit_regression=.01),
        acceptance_scope='Repeat the retained BOS/chat first-token checks and logit/FP64 comparisons; require at most one percentage point additional normalized RMS versus the qualified L40S result. This is a numerical regression gate, not a corpus accuracy evaluation.')
    recorder=Recorder('/data/standard-benchmarks',metadata,kind='qualification',
        parent=metadata['qualified_akas_run_id'],change='Qualify the H200 AOT program using the retained L40S independent references; upstream timing baselines remain frozen',
        dependency_sources={'tensor':Path('/workspace/tensor')})
    recorder.artifact(Path(__file__),'benchmark.py')
    for name in ['provenance.json','akas_results.json','vllm_results.json','sglang_results.json','fp64-reference.json']:
        recorder.artifact(refs/name,'references/'+name)
    log=[];cases=[];logits={};status='failed'
    def emit(*items):
        line=' '.join(map(str,items));log.append(line);print(line,flush=True)
    def nrms(a,b):
        return float(np.linalg.norm(a.astype('float64')-b)/max(float(np.linalg.norm(b)),1e-20))
    emit('STARTED',recorder.id,program,metadata['program_sha256'])
    try:
        with tensor.Device() as device, LFM2MoE('/data/checkpoint',bundle,device,
                checkpoint_options={'tokenizer':False}) as model:
            metadata['device']=device.info
            model.prefill_group_size=16
            for suite in ['nobos','bos','chat']:
                for index,prompt in enumerate(old[suite]):
                    key=f'{suite}_{index}'
                    for runtime in results:
                        if prompt['tokens']!=results[runtime][suite][index]['tokens']:
                            raise ValueError('qualification input token mismatch')
                    with Session(model,batch=1,chunk=2048) as session:
                        session.reset();first=session.prefill([prompt['tokens']])[0]
                        actual=session.read_feature('logits')[0].copy()
                        ids=[int(first)]
                        for _ in range(31):ids.append(int(session.decode(1)[0]))
                    comparisons={}
                    for runtime in ['vllm','sglang']:
                        reference=arrays[runtime][key]
                        comparisons[runtime]=dict(rms=float(np.sqrt(np.mean((actual-reference)**2))),
                            normalized_rms=nrms(actual,reference),
                            l40s_normalized_rms=nrms(arrays['akas'][key],reference),
                            first_token_equal=first==results[runtime][suite][index]['first_token'])
                    if suite=='nobos':
                        if not np.array_equal(prompt['tokens'],fp64[f'tokens{index}']):
                            raise ValueError('FP64 reference input token mismatch')
                        reference=fp64[f'logits{index}']
                        comparisons['fp64']=dict(rms=float(np.sqrt(np.mean((actual-reference)**2))),
                            normalized_rms=nrms(actual,reference),
                            l40s_normalized_rms=nrms(arrays['akas'][key],reference))
                    finite=bool(np.isfinite(actual).all())
                    regression=all(c['normalized_rms']<=c['l40s_normalized_rms']+.01 for c in comparisons.values())
                    first_ok=suite=='nobos' or all(c['first_token_equal'] for c in comparisons.values())
                    case=dict(suite=suite,index=index,input_ids=prompt['tokens'],output_ids=ids,
                        first_token=int(first),finite=finite,comparisons=comparisons,
                        passed=bool(finite and regression and first_ok),
                        normalized_rms_vs_l40s_akas=nrms(actual,arrays['akas'][key]))
                    cases.append(case);logits[key]=actual
                    emit('NUMERICAL',suite,index,'token',first,'passed',case['passed'],
                         'vs_L40S',round(case['normalized_rms_vs_l40s_akas'],6),
                         'reference_agreement',[c.get('first_token_equal') for c in comparisons.values()])
                    (recorder.directory/'numerical.json').write_text(json.dumps(dict(metadata=metadata,cases=cases),indent=2))
            recorder.program(bundle)
        status='passed' if all(c['passed'] for c in cases) and len(cases)==24 else 'failed'
    finally:
        metadata.update(qualification_status=status,summary={s:dict(
            cases=sum(c['suite']==s for c in cases),passed=sum(c['suite']==s and c['passed'] for c in cases),
            three_way_first_token_agreement=sum(c['suite']==s and all(c['comparisons'][r]['first_token_equal'] for r in ['vllm','sglang']) for c in cases)) for s in ['nobos','bos','chat']})
        raw=recorder.directory/'numerical.json';raw.write_text(json.dumps(dict(metadata=metadata,cases=cases),indent=2))
        saved=recorder.directory/'akas_logits.npz';np.savez_compressed(saved,**logits)
        worker=recorder.directory/'worker.log';worker.write_text('\n'.join(log)+'\n')
        for path in [raw,saved,worker]:recorder.artifact(path,path.name)
        record=recorder.update(dict(metadata=metadata,results=[]),final=True,status=status)
        volume.commit();temporary.cleanup()
    emit('QUALIFICATION',recorder.id,record['status'],metadata['summary'])
    return dict(run_id=recorder.id,status=record['status'],summary=metadata['summary'])


@app.local_entrypoint()
def main(action: str='numerical',program: str='optimized-v5'):
    import json,time
    if action!='numerical':raise ValueError('unsupported qualification action')
    ledger_path=ROOT/'build/h200/cost-ledger.json';ledger=json.loads(ledger_path.read_text())
    rate=.001261+4*.0000131+64*.00000222
    if sum(r['conservative_usd'] for r in ledger)+600*rate>28:
        raise RuntimeError('insufficient remaining budget')
    row=dict(action='akas-independent-numerical',program=program,status='reserved',started_at=time.time(),conservative_usd=600*rate)
    ledger.append(row);ledger_path.write_text(json.dumps(ledger,indent=2));tick=time.monotonic()
    try:
        result=numerical.remote(program);row.update(status='completed',result=result);print(json.dumps(result),flush=True)
    finally:
        row['elapsed_seconds']=time.monotonic()-tick
        if row['status']=='completed':row['conservative_usd']=(row['elapsed_seconds']+30)*rate
        ledger_path.write_text(json.dumps(ledger,indent=2))

def phase_compare(program: str):
    import json
    import time
    import hashlib
    import numpy as np
    import tensor
    from akas.models.lfm2 import LFM2MoE
    from akas.benchmarking.session import Session
    from tools.baselines.common import synthetic_prompts
    from tools.h200_native import elapsed

    out=Path('/data/results')/f'akas-phase-compare-{time.time_ns()}.json'
    report=dict(protocol='native-gpu-prefill-ar127-v1',reference='final-best',candidate=program,cases=[])
    references={}
    with tensor.Device() as device:
        for name in ['final-best',*program.split(',')]:
            path=Path('/data/programs')/name
            with LFM2MoE('/data/checkpoint',path,device,checkpoint_options={'tokenizer':False}) as model:
                for batch,group in ([(256,4)] if name=='final-best' else [(256,4),(256,16)]):
                    model.prefill_group_size=group
                    prompts=synthetic_prompts(model.config.vocab,batch,2048)
                    case=dict(program=name,batch=batch,group=group,program_sha256=hashlib.sha256((path/'program.json').read_bytes()).hexdigest(),samples=[])
                    report['cases'].append(case)
                    for repeat in range(3):
                        with Session(model,batch=batch,chunk=2048) as session:
                            session.reset()
                            first,prefill=elapsed(device,lambda:session.prefill(prompts))
                            logits=session.read_feature('logits').astype('float32')
                            # The last native prefill group owns this diagnostic buffer.
                            reference_key=(batch,group)
                            if name=='final-best':
                                references[(batch,4)]=(logits.copy(),list(first))
                            ref,tokens=references[(batch,4)]
                            checked=min(len(logits),len(ref))
                            actual,expected=logits[-checked:],ref[-checked:]
                            nrms=float(np.sqrt(np.mean((actual-expected)**2))/(np.sqrt(np.mean(expected**2))+1e-20))
                            agreement=sum(x==y for x,y in zip(first,tokens))/batch
                            valid=bool(np.isfinite(logits).all() and nrms<.01 and agreement>=.99)
                            if not valid:
                                case.update(status='failed',normalized_logit_rms=nrms,first_token_agreement=agreement)
                                out.write_text(json.dumps(report,indent=2));volume.commit()
                                raise RuntimeError('candidate model prefill qualification failed')
                            _,decode=elapsed(device,lambda:session.decode(127,read=False))
                            last=session.executor.last_frame.buffers[session.executor.last_frame.prediction].to_numpy()[:batch].astype(int).tolist()
                            row=dict(prefill_seconds=prefill,decode_seconds=decode,
                                prefill_tokens_per_second=batch*2048/prefill,decode_tokens_per_second=batch*127/decode,
                                normalized_logit_rms=nrms,checked_logit_rows=checked,
                                first_token_agreement=agreement,first_tokens=first,last_tokens=last)
                            if repeat:case['samples'].append(row)
                            print('PHASE COMPARE',name,batch,group,repeat,round(row['prefill_tokens_per_second']),round(row['decode_tokens_per_second']),'NRMS',nrms,'agreement',agreement,flush=True)
                        out.write_text(json.dumps(report,indent=2));volume.commit()
                    case['status']='passed'
    report['status']='passed';out.write_text(json.dumps(report,indent=2));volume.commit()
    return dict(out=str(out),status=report['status'])

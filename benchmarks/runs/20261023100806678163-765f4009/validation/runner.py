"""Bounded real-checkpoint DSpark compilation, numerical and serving qualification."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
import tensor
from tensor.compiler.search import ScheduleProfile
from akas.artifacts.read import read_program
from akas.benchmarking.history import Recorder, write
from akas.compilation.compile import compile_program
from akas.compilation.policy import lower
from akas.compilation.tuning import SearchSettings
from akas.config import EngineConfig
from akas.engine import Engine
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.lfm2.model import LFM2MoE
from akas.models.dspark import DSparkDraft, DSparkProposer
from akas.models.dspark.weights import DSparkDraftCheckpoint

ROOT = Path('/home/sagemaker-user/akas')
OUT = ROOT / 'build/lfm-moe/dspark-cuda-smoke'
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = ROOT / 'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
DRAFT = CHECKPOINT.with_name('dspark-f16.gguf')
TARGET_PROGRAM, DRAFT_PROGRAM = OUT/'target', OUT/'draft'
loaded = LFM2MoE.checkpoint_type(CHECKPOINT)
sidecar = DSparkDraftCheckpoint(DRAFT, target=loaded)
POLICY = dict(attention=dict(splits=64, decoded_kv=False), moe=dict(fused_swiglu=True))
metadata = dict(runtime='akas', protocol='composition-and-serving-check-v1',
    purpose='validation', provider='cuda', target='sm_89',
    model='akas.models.lfm2.model:LFM2MoE', mode='dspark', kv_dtype='float16',
    checkpoint=str(CHECKPOINT), checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
    draft_checkpoint=str(DRAFT), draft_checkpoint_sha256='0cd847ca049aeabf53cb41d62091ae20eb50245cf89594c75af204784a1b1a0f',
    max_context=4096, chunk=32, window=10, block_size=9, confidence_threshold=0,
    shape_timeout_seconds=120, warmup=0, repeats=1,
    target_layers=list(sidecar.config.target_layers), model_throughput_qualified=False,
    qualification='Independent operation references, bounded full-draft NumPy reference and target-verified serving; this is not a throughput benchmark')
rec = Recorder(ROOT/'benchmarks', metadata, kind='algorithm',
    parent='20261023072001816909-fb5b388f',
    change='Compose native F16 DSpark sidecar and target convolution verification with device-resident feature handoff')
report = dict(schema='akas.benchmark.v1', metadata=metadata, results=[])
write(OUT/'run-id.json', dict(run_id=rec.id))
rec.update(report)
print('RUN', rec.id, flush=True)
settings = SearchSettings(budget=1, width=1, warmup=0, samples=1, repeats=1)

def normalized(p):
    p = json.loads(json.dumps(p))
    for name in ('capacity', 'staged_kv', 'position_limit'):
        p.pop(name, None)
    return json.dumps(p, sort_keys=True)

source_entries = []
previous = ROOT/'build/lfm-moe/cuda-f16-full/cuda/program'
for phase in read_program(previous)['phases']:
    manifest = json.loads((previous/phase['path']).read_text())
    source_entries.extend(manifest['tuning']['profile']['entries'])
lookup = {(e['operation'],normalized(e['parameters'])): e['schedule'] for e in source_entries}
entries = {}
for chunk, phase in ((1,None), (32,None), (10,'verify')):
    composed = LFM2MoE.compose(loaded, rows=1, chunk=chunk, max_requests=1,
        max_context=4096, provider='cuda', phase=phase, kv_dtype='float16',
        auxiliary_layers=sidecar.config.target_layers)
    physical, _ = lower(composed.graph, composed.commit, policy=POLICY,
                         provider='cuda', target='sm_89')
    for call in physical.calls:
        axes = lookup.get((call.kind, normalized(call.parameters)))
        if call.kind == 'attention':
            axes = dict(implementation='attention.cuda.pipelined')
        if axes:
            entries[json.dumps([call.kind,call.parameters], sort_keys=True)] = dict(
                operation=call.kind, parameters=call.parameters, schedule=axes)
seed = ScheduleProfile(dict(schema='tensor.schedule-profile.v1',provider='cuda',
                            target='sm_89',entries=list(entries.values())))
write(OUT/'target-seed.json',seed.data)
passed = False
try:
    with tensor.Device(provider='cuda') as device:
        for label, reader, definition, geometries, options, policy, schedules in (
            ('draft', sidecar, DSparkDraft.compose,
             [(1,1,'context'), (1,10,'context'), (1,32,'context'), (1,9,'proposal')],
             dict(kv_dtype='float16'), None, None),
            ('target', loaded, LFM2MoE.compose, [(1,1), (1,32), (1,10,'verify')],
             dict(kv_dtype='float16', auxiliary_layers=sidecar.config.target_layers), POLICY, seed)):
            program = DRAFT_PROGRAM if label == 'draft' else TARGET_PROGRAM
            if os.environ.get('AKAS_DSPARK_REUSE_QUALIFIED') == '1':
                read_program(program)
                metadata['qualified_program_run_id'] = '20261023100310831676-a89a1684'
                rec.program(program)
                print('REUSE_QUALIFIED',label,flush=True)
                continue
            start = time.monotonic()
            print('COMPILE', label, flush=True)
            compile_program(reader, program, definition=definition, geometries=geometries,
                max_requests=1, max_context=4096, provider='cuda', target='sm_89',
                model_options=options, policy=policy, schedules=schedules,
                search_settings=settings, device=device)
            metadata[label+'_compilation'] = dict(seconds=time.monotonic()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            rec.program(program)
            rec.update(report)
            print('COMPILED', label, json.dumps(metadata[label+'_compilation']), flush=True)
        with ExitStack() as stack:
            target = stack.enter_context(LFM2MoE(CHECKPOINT,TARGET_PROGRAM,device,graphs=True))
            draft = stack.enter_context(DSparkDraft(DRAFT,DRAFT_PROGRAM,device,graphs=True,
                                                     checkpoint_options=dict(target=target)))
            from dspark_reference import qualify
            metadata['full_draft_reference'] = qualify(target,draft)
            print('DRAFT_REFERENCE',json.dumps(metadata['full_draft_reference']),flush=True)
            proposer = DSparkProposer(target,draft)
            decoder = SpeculativeDecoder(target,proposer,window=10)
            prompts = [loaded.tokenizer.encode(text) for text in (
                'List three ways to save water at home.',
                'What is the next integer after 41? Answer briefly.',
                'Explain in one paragraph why bridges need expansion joints.')]
            # Exercise multiple prefill chunks, both noise repair and slot reuse.
            prompts.append(prompts[0]*7)
            ar_outputs = []
            for prompt in prompts:
                slot = target.admit()
                output = target.prefill([slot], [prompt], chunk=32)
                for _ in range(47):
                    output.extend(target.decode([slot], [output[-1]]))
                target.release(slot)
                ar_outputs.append(output)
            with Engine(decoder,config=EngineConfig(prefill_chunk=32,prefill_requests=1)) as engine:
                for prompt, ar_output in zip(prompts, ar_outputs):
                    before=dict(decoder.stats)
                    request=engine.submit(prompt,max_tokens=48)
                    start=time.monotonic()
                    while engine.step():
                        if time.monotonic()-start>120:
                            raise TimeoutError('DSpark serving shape exceeded 120 seconds')
                    device.synchronize()
                    stats={key:decoder.stats[key]-before[key] for key in before}
                    row=dict(status='ok',case=dict(batch=1,prompt=len(prompt),generated=48),
                        output_ids=[request.output], ar_output_ids=[ar_output],
                        ar_exact=request.output==ar_output,
                        ar_token_agreement=sum(a==b for a,b in zip(request.output,ar_output))/48,
                        seconds=time.monotonic()-start,
                        speculation=stats,correctness=dict(correct=True,
                            scope='Greedy target verification and accepted-prefix state commit; full draft checked independently'))
                    if len(request.output)!=48 or target.occupied.any() or draft.occupied.any():
                        raise ValueError('Incomplete output or leaked request slot')
                    report['results'].append(row)
                    rec.update(report)
                    print('SERVED',json.dumps(dict(prefix=len(prompt),seconds=row['seconds'],stats=stats)),flush=True)
    passed=True
except BaseException as error:
    metadata['error']=f'{type(error).__name__}: {error}'
    metadata['traceback']=traceback.format_exc()
    print(metadata['traceback'],flush=True)
finally:
    metadata['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for program in (TARGET_PROGRAM,DRAFT_PROGRAM):
        if (program/'program.json').is_file():
            rec.program(program)
    for source,name in ((Path(__file__),'validation/runner.py'),
                         (Path(__file__).with_name('dspark_reference.py'),'validation/reference.py'),
                         (OUT/'target-seed.json','validation/target-seed.json')):
        if source.is_file(): rec.artifact(source,name)
    write(OUT/'report.json',report)
    rec.update(report,final=True,status='passed' if passed else 'failed')
    print('RETAINED',rec.id,passed,flush=True)
if not passed: raise SystemExit(1)

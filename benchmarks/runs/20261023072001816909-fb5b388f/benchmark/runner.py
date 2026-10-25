"""Full CUDA demo workload, retained through the standard benchmark recorder."""
import asyncio
from contextlib import contextmanager
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
from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder, write
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.compilation.compile import compile_program
from akas.compilation.policy import lower
from akas.compilation.tuning import SearchSettings
from akas.config import EngineConfig
from akas.engine import Engine
from akas.models.lfm2.model import LFM2MoE
from akas.optimization.evaluation import http_benchmark, measured_requests, prompts
from akas.serving.http import Service
from kv_quality import collect

ROOT = Path('build/lfm-moe/cuda-f16-full/cuda')
ROOT.mkdir(parents=True, exist_ok=True)
PROGRAM = ROOT / 'program'
PREVIOUS = Path('build/lfm-moe/cuda-ordered-full/cuda/program')
PARENT = '20261020211932758011-47b0b378'
REFERENCE = json.loads((Path('benchmarks/runs') / PARENT / 'report.json').read_text())
PREFIXES = (32, 512, 2048, 8192, 16384, 32768, 65536, 130944)
CHECKPOINT = Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve()
loaded = LFM2MoE.checkpoint_type(CHECKPOINT)
policy = dict(attention=dict(splits=64, decoded_kv=False), moe=dict(fused_swiglu=True))
source_entries = []
for phase in read_program(PREVIOUS)['phases']:
    manifest = json.loads((PREVIOUS / phase['path']).read_text())
    source_entries.extend(manifest['tuning']['profile']['entries'])
def normalized(parameters):
    parameters = json.loads(json.dumps(parameters))
    parameters.pop('staged_kv', None)
    if 'kv_dtype' in parameters:
        parameters['kv_dtype'] = 'float16'
    if 'unit_scales' in parameters:
        parameters['unit_scales'] = True
    return json.dumps(parameters, sort_keys=True)
lookup = {(e['operation'], normalized(e['parameters'])): e['schedule'] for e in source_entries}
entries = {}
unmatched = set()
for chunk in (1, 32, 512, 2048, 4096):
    composed = LFM2MoE.compose(loaded, rows=1, chunk=chunk, max_requests=1,
        max_context=131072, provider='cuda', kv_dtype='float16', optimized=True)
    physical, _ = lower(composed.graph, composed.commit, policy=policy,
        provider='cuda', target='sm_89')
    for call in physical.calls:
        kind, parameters = call.kind, call.parameters
        axes = lookup.get((kind, normalized(parameters)))
        if kind == 'attention':
            axes = dict(implementation='attention.cuda.pipelined')
        if axes is None:
            unmatched.add(kind)
            continue
        entry = dict(operation=kind, parameters=parameters, schedule=axes)
        entries[json.dumps([kind, parameters], sort_keys=True)] = entry
    del composed, physical
print('SEED', len(entries), 'DEFAULT_FAMILIES', sorted(unmatched), flush=True)
seed = ScheduleProfile(dict(schema='tensor.schedule-profile.v1', provider='cuda',
    target='sm_89', entries=list(entries.values())))
write(ROOT / 'seed-profile.json', seed.data)
meta = dict(runtime='akas', purpose='benchmark', protocol='native-token-step-v2',
    provider='cuda', target='sm_89', device=dict(name='NVIDIA L4', provider='cuda'),
    model='akas.models.lfm2.model:LFM2MoE', definition='akas.models.lfm2.model:LFM2MoE',
    checkpoint=str(CHECKPOINT),
    checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
    weight_format='q4_0', weight_dtype='gguf-q4_0-mixed', activation_dtype='float32',
    kv_dtype='float16', mode='ar', chunk=4096, graphs=True, warmup=1, repeats=3,
    output_tokens=128, ignore_eos=True, policy=policy, shape_timeout_seconds=120,
    cpu_reference_threads=4, prompt_generation='17 + batch * 4099 + prefix_tokens',
    requested_prefixes=list(PREFIXES), fp8_comparison_run_id=PARENT,
    precision_reference_run_id='20261020134751392443-6b1662b8',
    qualification='Independent codec/operation references and exact 128-token native/Engine/HTTP consistency on every shape; F16 KV for matched llama.cpp comparison. Precision drift is measured separately, not a corpus accuracy claim.',
    kernel_evidence_runs=['20261020205941837023-84bc5997', '20261020210651016012-a3613aec', '20261020210822967727-b12a9cf7', '20261020211643710127-7d56732c'],
    performance_target=dict(prefill_speedup=2, decode_speedup=1.5,
        baselines=['20261018202355478915-6adad97d', '20261018203703691867-28072a0a'],
        precision_comparison='F16 KV for both AKAS and llama.cpp; FP8 comparison reported separately',
        phase_comparison='AKAS synchronized native phases versus llama.cpp server eval phases; HTTP measurements reported separately'))
rec = Recorder('benchmarks', meta, kind='algorithm', parent=PARENT,
    change='Matched F16 KV CUDA benchmark with fused expert decode, retained projection geometry and unpadded medium prompts; exact native/Engine/HTTP outputs')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
write(ROOT / 'run-id.json', dict(run_id=rec.id))
rec.update(report)
passed = False

def checkpoint_report():
    write(ROOT / 'report.json', report)
    rec.update(report)

@contextmanager
def factory():
    with tensor.Device(provider='cuda') as device:
        with LFM2MoE(CHECKPOINT, PROGRAM, device, graphs=True) as model:
            with Engine(model, config=EngineConfig(prefill_chunk=4096, prefill_requests=1)) as engine:
                yield Service(engine, model.checkpoint.tokenizer.encode,
                              dict(model=meta['model'], provider='cuda', kv_dtype='float16'))

try:
    print('START', rec.id, 'PREFIXES', list(PREFIXES), flush=True)
    started = time.monotonic()
    with tensor.Device(provider='cuda') as device:
        compile_program(loaded, PROGRAM, definition=LFM2MoE.compose,
            geometries=[(1, 1), (1, 32), (1, 512), (1, 2048), (1, 4096)], max_requests=1, max_context=131072,
            provider='cuda', target='sm_89', device=device,
            model_options=dict(kv_dtype='float16', optimized=True), schedules=seed,
            policy=policy,
            search_settings=SearchSettings(budget=1, width=1, warmup=0, samples=1, repeats=1))
        meta['compilation_resources'] = dict(seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        print('COMPILED', json.dumps(meta['compilation_resources']), flush=True)
        checkpoint_report()
        with LFM2MoE(CHECKPOINT, PROGRAM, device, graphs=True) as model:
            values, checks = collect(model, loaded.tokenizer, chunk=4096)
            np.savez_compressed(ROOT / 'precision-f16.npz', **values)
            meta['precision_probe'] = checks
            with Engine(model, config=EngineConfig(prefill_chunk=4096, prefill_requests=1)) as engine:
                for prefix in PREFIXES:
                    print('MEASURE', prefix, flush=True)
                    with Session(model, batch=1, chunk=4096, timeout=120) as session:
                        row = run(session, Case(1, prefix, 128), warmup=1, repeats=3)
                        session.reset()
                        native_ids = list(session.prefill(prompts(model.config.vocab, 1, prefix)))
                        for _ in range(127):
                            native_ids.extend(session.decode(1))
                    row['native_output_ids'] = [native_ids]
                    report['results'].append(row)
                    checkpoint_report()
                    samples = []
                    output = None
                    for iteration in range(4):
                        measured = measured_requests(engine, 1, prefix, 128, timeout=120)
                        ids = measured.pop('output_ids')
                        if ids != [native_ids] or (output is not None and ids != output):
                            raise ValueError(f'Native/Engine output differs at {prefix}')
                        output = ids
                        if iteration:
                            samples.append(measured)
                    row.update(output_ids=output, serving=dict(samples=samples),
                        correctness=dict(correct=True, scope='Exact native/Engine 128-token equality; HTTP checked after native matrix'))
                    print('RESULT', prefix, 'pp', row['summary']['prefill_tokens_per_second']['median'],
                          'tg', row['summary']['decode_tokens_per_second']['median'], 'correct', True, flush=True)
                    checkpoint_report()
    print('HTTP_ALL_CONTEXTS', flush=True)
    os.environ['AKAS_SHAPE_TIMEOUT'] = '120'
    os.environ['AKAS_BENCHMARK_WARMUP'] = '1'
    os.environ['AKAS_BENCHMARK_REPEATS'] = '3'
    if os.environ.get('AKAS_SKIP_AUTOTUNE') == '1':
        raise ValueError('Full benchmark cannot run with correctness-skipping smoke settings')
    meta['http'] = asyncio.run(http_benchmark(factory, report['results'], vocab=128000))
    for row in report['results']:
        row['correctness']['scope'] = meta['qualification']
    print('HTTP_PASSED', json.dumps(meta['http']), flush=True)
    passed = True
except BaseException as error:
    meta['error'] = f'{type(error).__name__}: {error}'
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    meta['cpu_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if (PROGRAM / 'program.json').exists():
        rec.program(PROGRAM)
    for path, name in ((Path(__file__), 'benchmark/runner.py'),
                       (ROOT / 'seed-profile.json', 'benchmark/seed-profile.json'),
                       (Path('build/lfm-moe/kv_quality.py'), 'precision/probe.py'),
                       (ROOT / 'precision-f16.npz', 'precision/f16.npz')):
        if path.exists():
            rec.artifact(path, name)
    for name in ('ordered-prefill-full-tests.log', 'public-ordered-attention-tests.log'):
        path = Path('build/lfm-moe') / name
        if path.exists():
            rec.artifact(path, 'validation/' + name)
    write(ROOT / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

# Publish the same report bundle used by ordinary matrix campaigns. This is one
# fixed incumbent qualification, with one candidate, rather than a tuning search.
from akas.optimization.context import identity, workload
from akas.optimization.deployment import describe
from tools.campaign_report import export

context = dict(checkpoint=str(CHECKPOINT), definition=meta['definition'],
    workload=workload('single-user-long-context-v1'), backend='cuda', device=0,
    memory_bytes=None, ttft_ms=None, tpot_ms=None, kv_dtype='float16', mode='ar')
recipe = dict(provider='cuda', target='sm_89', chunk=4096, prefill_requests=1,
    graphs=True, trials=1, policy=policy, model_options=dict(kv_dtype='float16', optimized=True),
    window=None, deferred=False, head=None)
recipe['id'] = identity(recipe)[:16]
campaign_identity = identity(dict(source=rec.record['source']['tree_sha256'],
    recipe=recipe, workload=context['workload'], run_id=rec.id))
deployment = describe(context, recipe, PROGRAM,
    checkpoint_sha256=meta['checkpoint_sha256'], campaign_identity=campaign_identity,
    run_id=rec.id, hardware=meta['device'])
deployment['status'] = 'qualified'
write(ROOT / 'deployment.json', deployment)
profile_id = 'lfm2.5-24b-a2b-q4_0-cuda-float16-ar'
campaign_root = ROOT.parent / 'campaign'
elapsed = (Path(ROOT / 'run-id.json').stat().st_mtime)
elapsed = time.time() - elapsed
entry = dict(id=profile_id, context=context, status='complete',
    deployment=str((ROOT / 'deployment.json').resolve()),
    winner=dict(run_id=rec.id, recipe=recipe['id']), spent_seconds=elapsed)
state = dict(schema='akas.matrix-campaign.v1', identity=campaign_identity,
    specification=dict(schema='akas.matrix-campaign.v1', models=['lfm2.5-24b-a2b'],
        formats=['q4_0'], backends=['cuda'], caches=['float16'],
        workload='single-user-long-context-v1', execution='fixed-incumbent-qualification'),
    models={'lfm2.5-24b-a2b': json.loads(Path('benchmarks/models/lfm2.5-24b-a2b.json').read_text())},
    source=rec.record['source'], created_at=rec.record['started_at'],
    updated_at=rec.record['completed_at'], status='complete', spent_seconds=elapsed,
    entries=[entry])
write(campaign_root / profile_id / 'campaign.json', dict(jobs=[dict(recipe=recipe,
    directory=str(ROOT.resolve()), run_id=rec.id, status='passed', elapsed_seconds=elapsed)]))
write(campaign_root / 'campaign.json', state)
destination = export(campaign_root / 'campaign.json', history='benchmarks')
write(ROOT / 'publication.json', dict(report=str(destination), deployment=str(ROOT / 'deployment.json')))
print('PUBLISHED', destination, flush=True)

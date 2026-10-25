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

ROOT = Path('build/lfm-moe/cuda-full/cuda')
ROOT.mkdir(parents=True, exist_ok=True)
PROGRAM = ROOT / 'program'
PREVIOUS = Path('build/lfm-moe/staged-model/cuda/program')
PARENT = '20261020172334890603-b8fd2763'
REFERENCE = json.loads((Path('benchmarks/runs') / PARENT / 'report.json').read_text())
PREFIXES = (32, 512, 2048, 8192, 16384, 32768, 65536, 130944)
CHECKPOINT = Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve()
loaded = LFM2MoE.checkpoint_type(CHECKPOINT)
policy = dict(attention=dict(splits=64, decoded_kv=True), moe=dict(fused_swiglu=True))
entries = {}
for phase in read_program(PREVIOUS)['phases']:
    manifest = json.loads((PREVIOUS / phase['path']).read_text())
    for original in manifest['tuning']['profile']['entries']:
        entry = json.loads(json.dumps(original))
        entries[json.dumps([entry['operation'], entry['parameters']], sort_keys=True)] = entry
composition = LFM2MoE.compose(loaded, rows=1, chunk=1, max_requests=1,
                            max_context=131072, provider='cuda', kv_dtype='fp8', optimized=True)
graph, _ = lower(composition.graph, composition.commit, policy=policy,
                 provider='cuda', target='sm_89')
fused = [c for c in graph.calls if c.kind == 'expert_swiglu']
if len(fused) != 38:
    raise ValueError(f'Expected 38 expert decode fusions, found {len(fused)}')
for call in fused:
    entry = dict(operation=call.kind, parameters=call.parameters,
                 schedule=dict(implementation='moe.cuda.packed_swiglu', threads=128,
                               columns_per_warp=1, unroll=2))
    entries[json.dumps([entry['operation'], entry['parameters']], sort_keys=True)] = entry
del composition, graph, fused
seed = ScheduleProfile(dict(schema='tensor.schedule-profile.v1', provider='cuda',
                            target='sm_89', entries=list(entries.values())))
write(ROOT / 'seed-profile.json', seed.data)
meta = dict(runtime='akas', purpose='benchmark', protocol='native-token-step-v2',
    provider='cuda', target='sm_89', device=dict(name='NVIDIA L4', provider='cuda'),
    model='akas.models.lfm2.model:LFM2MoE', definition='akas.models.lfm2.model:LFM2MoE',
    checkpoint=str(CHECKPOINT),
    checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
    weight_format='q4_0', weight_dtype='gguf-q4_0-mixed', activation_dtype='float32',
    kv_dtype='fp8', mode='ar', chunk=4096, graphs=True, warmup=1, repeats=3,
    output_tokens=128, ignore_eos=True, policy=policy, shape_timeout_seconds=120,
    cpu_reference_threads=4, prompt_generation='17 + batch * 4099 + prefix_tokens',
    requested_prefixes=list(PREFIXES), same_precision_reference_run_id=PARENT,
    precision_reference_run_id='20261020134751392443-6b1662b8',
    qualification='Independent codec/operation references; exact 128-token native/Engine/HTTP consistency on every shape; unchanged FP8 model reference at 32, 8192 and 130944; teacher-forced F16 drift probe at those three boundaries, not corpus accuracy.',
    kernel_evidence_runs=['20261020174844718413-384c5b14', '20261020174911320354-58e32599'],
    performance_target=dict(prefill_speedup=2, decode_speedup=1.5,
        baselines=['20261018202355478915-6adad97d', '20261018203703691867-28072a0a'],
        precision_comparison='FP8 KV versus F16 KV, reported separately',
        phase_comparison='AKAS synchronized native phases versus llama.cpp server eval phases; HTTP measurements reported separately'))
rec = Recorder('benchmarks', meta, kind='algorithm', parent=PARENT,
    change='Full eight-context CUDA benchmark with compiler-owned FP8 prefill staging and exact fused expert decode; shape-matched native/Engine/HTTP prompts')
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
                              dict(model=meta['model'], provider='cuda', kv_dtype='fp8'))

try:
    print('START', rec.id, 'PREFIXES', list(PREFIXES), flush=True)
    started = time.monotonic()
    with tensor.Device(provider='cuda') as device:
        compile_program(loaded, PROGRAM, definition=LFM2MoE.compose,
            geometries=[(1, 1), (1, 32), (1, 4096)], max_requests=1, max_context=131072,
            provider='cuda', target='sm_89', device=device,
            model_options=dict(kv_dtype='fp8', optimized=True), schedules=seed,
            policy=policy,
            search_settings=SearchSettings(budget=1, width=1, warmup=0, samples=1, repeats=1))
        meta['compilation_resources'] = dict(seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        print('COMPILED', json.dumps(meta['compilation_resources']), flush=True)
        checkpoint_report()
        with LFM2MoE(CHECKPOINT, PROGRAM, device, graphs=True) as model:
            with np.load('build/lfm-moe/fp8-model/cuda/precision-f16.npz', allow_pickle=False) as f16:
                values, checks = collect(model, loaded.tokenizer, chunk=4096, reference=f16)
            np.savez_compressed(ROOT / 'precision-fp8.npz', **values)
            meta['precision_comparison'] = checks
            with np.load('build/lfm-moe/staged-model/cuda/precision-fp8.npz', allow_pickle=False) as old:
                agreement = [dict(name=name, bitwise=bool(np.array_equal(value, old[name])),
                    max_absolute_error=float(np.max(np.abs(value.astype('float64') - old[name]))))
                    for name, value in values.items()]
            meta['same_precision_logit_comparison'] = agreement
            print('SAME_PRECISION', json.dumps(agreement), flush=True)
            if not all(item['bitwise'] for item in agreement):
                raise ValueError('Fused expert decode changes qualified FP8 precision-probe values')
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
                    previous = next((r for r in REFERENCE['results'] if r['case']['prompt'] == prefix), None)
                    if previous is not None:
                        same = output == previous['output_ids']
                        row['native_correctness'] = dict(correct=same, reference_run_id=PARENT)
                        if not same:
                            row['correctness']['correct'] = False
                            raise ValueError(f'Fused CUDA output differs from FP8 reference at {prefix}')
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
                       (ROOT / 'precision-fp8.npz', 'precision/fp8.npz')):
        if path.exists():
            rec.artifact(path, name)
    for name in ('moe-fusion-full-tests.log', 'moe-fusion-worktree-gpu-tests.log',
                 'moe-fusion-worktree-final-cpu-tests.log'):
        path = Path('build/lfm-moe') / name
        if path.exists():
            rec.artifact(path, 'validation/' + name)
    write(ROOT / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

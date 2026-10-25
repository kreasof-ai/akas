"""Qualify source-integrated portable decode through compile, native, Engine and HTTP."""
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
from akas.optimization.planning import estimate
from akas.serving.http import Service

root = Path('build/lfm-moe/public-webgpu-decode')
root.mkdir(exist_ok=True)
program = root / 'program'
old = Path('build/lfm-moe/manual-v2-matched/webgpu/program')
reference = json.loads((old.parent / 'report.json').read_text())
checkpoint = Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf').resolve()
loaded = LFM2MoE.checkpoint_type(checkpoint)
policy = dict(vocabulary=dict(parallel_argmax=True, block=1024))
entries = {}
for phase in read_program(old)['phases']:
    manifest = json.loads((old / phase['path']).read_text())
    for original in manifest['tuning']['profile']['entries']:
        e = json.loads(json.dumps(original))
        p = e['parameters']
        if e['operation'] == 'linear' and not p.get('ggml', False) and p['r'] == 1 and p['k'] >= 128 and p['o'] <= 256:
            e['schedule'] = dict(implementation='linear.webgpu.column_projection', threads=256)
        if e['operation'] == 'quantized_experts' and p['r'] == 1:
            assert p['round_inputs'] is False
            e['schedule'] = dict(implementation='moe.webgpu.packed_float_decode', columns_per_warp=1, unroll=1)
        entries[json.dumps([e['operation'], p], sort_keys=True)] = e
seed = ScheduleProfile(dict(schema='tensor.schedule-profile.v1', provider='webgpu', target='webgpu-portable-v1', entries=list(entries.values())))
write(root / 'seed-profile.json', seed.data)
needed = 0
for chunk in (1, 32, 512):
    composition = LFM2MoE.compose(loaded, rows=1, chunk=chunk, max_requests=1, max_context=131072, provider='webgpu', kv_dtype='float16', optimized=True)
    graph, _ = lower(composition.graph, composition.commit, policy=policy, provider='webgpu', target='webgpu-portable-v1')
    needed = max(needed, estimate(graph)['largest_buffer_bytes'])
del composition, graph
meta = dict(runtime='akas', purpose='benchmark', protocol='native-token-step-v2',
    provider='webgpu', target='webgpu-portable-v1', device=dict(name='NVIDIA L4', provider='webgpu'),
    model='akas.models.lfm2.model:LFM2MoE', definition='akas.models.lfm2.model:LFM2MoE', checkpoint=str(checkpoint),
    checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5',
    weight_format='q4_0', weight_dtype='gguf-q4_0-mixed', activation_dtype='float32', kv_dtype='float16', mode='ar',
    graphs=True, chunk=512, warmup=1, repeats=3, output_tokens=128, ignore_eos=True, requested_prefixes=[32, 8192],
    policy=policy, max_buffer_size=needed, shape_timeout_seconds=120,
    prompt_generation='17 + batch * 4099 + prefix_tokens',
    same_precision_reference_run_id='20261020191047151817-8e1d5140',
    kernel_evidence_runs=['20261020191837105510-7798741d','20261020193348533055-bc9b08e1','20261020195031655272-927c4e8a','20261020195237721793-a4187a4a'],
    qualification='Compiler-owned packed F32 expert products, column-parallel F32 router projections and two-stage vocabulary selection. F16 KV; no operand or accumulator precision reduction. Exact 128-token equality against retained F16 WebGPU output and between native, Engine and HTTP on both requested shapes. Short and 8K qualification only; no full-context WebGPU claim.')
rec = Recorder('benchmarks', meta, kind='algorithm', change='Qualify the integrated WebGPU decode compiler and production serving path', parent='20261020195237721793-a4187a4a')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False

def save():
    write(root / 'report.json', report)
    rec.update(report)

@contextmanager
def factory():
    with tensor.Device(provider='webgpu', max_buffer_size=needed) as device:
        with LFM2MoE(checkpoint, program, device, graphs=True) as model:
            with Engine(model, config=EngineConfig(prefill_chunk=512, prefill_requests=1)) as engine:
                yield Service(engine, model.checkpoint.tokenizer.encode, dict(model=meta['model'], provider='webgpu', kv_dtype='float16'))

try:
    print('START', rec.id, 'MAX_BUFFER', needed, flush=True)
    started = time.monotonic()
    with tensor.Device(provider='webgpu', max_buffer_size=needed) as device:
        compile_program(loaded, program, definition=LFM2MoE.compose, geometries=[(1,1),(1,32),(1,512)], max_requests=1, max_context=131072,
            provider='webgpu', target='webgpu-portable-v1', device=device, model_options=dict(kv_dtype='float16', optimized=True), schedules=seed, policy=policy,
            search_settings=SearchSettings(budget=1,width=1,warmup=0,samples=1,repeats=1))
        meta['compilation_resources'] = dict(seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        print('COMPILED', json.dumps(meta['compilation_resources']), flush=True)
        save()
        # Ensure the ordinary compiler actually selected the public implementations.
        phase = next(p for p in read_program(program)['phases'] if p['phase']=='decode')
        manifest = json.loads((program / phase['path']).read_text())
        selected = [e['schedule']['implementation'] for e in manifest['tuning']['profile']['entries']]
        for implementation in ('moe.webgpu.packed_float_decode','linear.webgpu.column_projection','akas.ops.vocabulary.webgpu:argmax_partial','akas.ops.vocabulary.webgpu:argmax_merge'):
            assert implementation in selected, (implementation, selected)
        meta['selected_decode_implementations'] = selected
        with LFM2MoE(checkpoint, program, device, graphs=True) as model:
            with Engine(model, config=EngineConfig(prefill_chunk=512, prefill_requests=1)) as engine:
                for prefix in (32,8192):
                    print('MEASURE', prefix, flush=True)
                    with Session(model, batch=1, chunk=512, timeout=120) as session:
                        row = run(session, Case(1,prefix,128), warmup=1,repeats=3)
                        session.reset()
                        ids = list(session.prefill(prompts(model.config.vocab,1,prefix)))
                        for _ in range(127):
                            ids.extend(session.decode(1))
                    expected = next(r for r in reference['results'] if r['case']['prompt']==prefix)['output_ids']
                    assert [ids] == expected, f'Native F16 reference mismatch at {prefix}'
                    row.update(native_output_ids=[ids], output_ids=[ids], correctness=dict(correct=True,scope=meta['qualification']))
                    report['results'].append(row)
                    save()
                    samples=[]
                    for iteration in range(4):
                        sample=measured_requests(engine,1,prefix,128,timeout=120)
                        assert sample.pop('output_ids')==[ids], f'Engine mismatch at {prefix}'
                        if iteration:
                            samples.append(sample)
                    row['serving']=dict(samples=samples)
                    print('RESULT',prefix,'pp',row['summary']['prefill_tokens_per_second']['median'],'tg',row['summary']['decode_tokens_per_second']['median'],'correct',True,flush=True)
                    save()
    print('HTTP',flush=True)
    os.environ.update(AKAS_SHAPE_TIMEOUT='120',AKAS_HTTP_TIMEOUT='120',AKAS_BENCHMARK_WARMUP='1',AKAS_BENCHMARK_REPEATS='3')
    assert os.environ.get('AKAS_SKIP_AUTOTUNE') != '1'
    meta['http']=asyncio.run(http_benchmark(factory,report['results'],vocab=128000))
    print('HTTP_PASSED',json.dumps(meta['http']),flush=True)
    passed=True
except BaseException:
    meta['traceback']=traceback.format_exc()
    print(meta['traceback'],flush=True)
finally:
    meta['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if (program/'program.json').exists():
        rec.program(program)
    for path,name in ((Path(__file__),'benchmark/runner.py'),(root/'seed-profile.json','benchmark/seed-profile.json'),(old.parent/'report.json','reference/webgpu-report.json')):
        rec.artifact(path,name)
    write(root/'report.json',report)
    rec.update(report,final=True,status='passed' if passed else 'failed')
    print('RETAINED',rec.id,passed,flush=True)
if not passed:
    raise SystemExit(1)

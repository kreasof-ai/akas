"""Separate unchanged short DSpark graph execution from host round coordination.

No kernels or serving behavior are changed. Synchronized phase attribution is
diagnostic; only the separate uninstrumented pass reports decode throughput.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
import ctypes as c
import json
import math
from pathlib import Path
import resource
import statistics
import time
import traceback

import numpy as np
import tensor
from tensor.compiler.tuning import measure_cuda

from akas.benchmarking.cases import Case
from akas.benchmarking.history import Recorder, digest, write
from akas.benchmarking.runner import run
from akas.benchmarking.speculative import SpeculativeSession
from akas.execution.forward import Forward
from akas.execution.plan import ExecutionPlan
from akas.generation.speculative.decoding import SpeculativeDecoder
from akas.models.dspark.proposer import load
from akas.models.lfm2.model import LFM2MoE

ROOT = Path('/home/sagemaker-user/akas')
OUT = ROOT / 'build/lfm-moe/dspark-short-latency'
OUT.mkdir(exist_ok=True)
TARGET = ROOT / 'build/lfm-moe/dspark-short-optimized/target'
DRAFT = ROOT / 'build/lfm-moe/dspark-short-q8/draft'
CHECKPOINT = ROOT / 'build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf'
WORKLOAD_PATH = ROOT / 'build/lfm-moe/dspark-short-q8/short-workload.json'
workload = json.loads(WORKLOAD_PATH.read_text())
previous = {}
for parent in ('20261023132226213745-5931c276', '20261023132331975932-f38d7467'):
    previous.update({row['text_case']['id']: row for row in
                     json.loads((ROOT/'benchmarks/runs'/parent/'report.json').read_text())['results']})
metadata = dict(
    runtime='akas', purpose='profiling', protocol='speculative-short-latency-v2',
    provider='cuda', target='sm_89', kv_dtype='float16', draft_kv_dtype='float16',
    draft_weight_dtype='q8_0', confidence_threshold=.5, proposal_tokens=9,
    verification_chunks=[1, 2, 4, 10], program_max_context=131072,
    shape_timeout_seconds=120, text_workload_sha256=digest(WORKLOAD_PATH),
    qualification='Ten fixed real-text prompts, 24-73 input tokens and 128 outputs; '
        'no long-context inference. Unchanged independently qualified programs. '
        'Uninstrumented throughput uses one warmup and three samples. '
        'Separate synchronized phase attribution adds boundaries and CUDA events; '
        'its residual is not a prediction of full-round capture speedup. '
        'Fixed-input verifier graph replay is a diagnostic, not generation. '
        'All diagnostic generations and verifier replays must match their '
        'uninstrumented speculative outputs exactly. No task accuracy claim.',
)
rec = Recorder(ROOT/'benchmarks', metadata, kind='algorithm',
               parent='20261023132331975932-f38d7467',
               change='Verify short DSpark GPU verifier latency separately from host coordination')
report = dict(schema='akas.benchmark.v1', metadata=metadata, results=[])
write(OUT/'run-id.json', dict(run_id=rec.id))
print('RUN', rec.id, flush=True)
passed = False
original_forward = Forward.run
try:
    with tensor.Device(provider='cuda') as device, ExitStack() as owned:
        target = owned.enter_context(LFM2MoE(CHECKPOINT, TARGET, device, graphs=True))
        head = load(target, checkpoint=CHECKPOINT.with_name('dspark-q8_0.gguf'),
                    bundle=DRAFT, graphs=True, confidence_threshold=.5)
        owned.callback(head.close)
        decoder = SpeculativeDecoder(target, head, window=10, capture=False)
        phase_samples = defaultdict(list)
        active = None
        events = []
        forwards = []
        micro_done = set()
        mode = 'none'
        micro_rows = []
        originals = {}

        def instrument_forward(frame):
            if active is None:
                return original_forward(frame)
            begin, end = c.c_void_p(), c.c_void_p()
            device.driver.call('cuEventCreate', c.byref(begin), 0)
            device.driver.call('cuEventCreate', c.byref(end), 0)
            events.append((begin, end))
            start = time.perf_counter()
            device.driver.call('cuEventRecord', begin, device.stream)
            value = original_forward(frame)
            device.driver.call('cuEventRecord', end, device.stream)
            forwards.append(dict(begin=begin, end=end,
                                 launch_wall_ms=(time.perf_counter()-start)*1000,
                                 captured=frame.graph is not None,
                                 operation_count=len(frame.plan)))
            return value

        def flush_events():
            rows = []
            for item in forwards:
                milliseconds = c.c_float()
                device.driver.call('cuEventElapsedTime', c.byref(milliseconds),
                                   item['begin'], item['end'])
                rows.append({k: v for k, v in item.items() if k not in ('begin', 'end')}
                            | dict(gpu_span_ms=milliseconds.value))
            for begin, end in events:
                device.driver.call('cuEventDestroy_v2', end)
                device.driver.call('cuEventDestroy_v2', begin)
            events.clear()
            forwards.clear()
            return rows

        def make_phase(function, label):
            def observed(*args, **kwargs):
                global active
                if mode != 'phases':
                    return function(*args, **kwargs)
                device.synchronize()
                active = label
                start = time.perf_counter()
                try:
                    value = function(*args, **kwargs)
                    device.synchronize()
                    elapsed = (time.perf_counter()-start)*1000
                    data = dict(wall_ms=elapsed, forwards=flush_events())
                    if label == 'verify':
                        data['chunk'] = kwargs['chunk']
                        data['live_tokens'] = len(args[1][0])
                    phase_samples[label].append(data)
                    return value
                finally:
                    active = None
            return observed

        def verifier_micro(*args, **kwargs):
            result = originals['verify'](*args, **kwargs)
            chunk = kwargs['chunk']
            if mode != 'micro' or chunk in micro_done:
                return result
            frame = target.batches.select(1, chunk, phase='verify')
            if frame.forward.graph is None:
                raise ValueError('Verifier is not captured')
            # Verify writes temporary KV/journals at unchanged positions. It
            # does not commit recurrent state; replay the same live input only.
            resources = tuple(frame.forward.buffers.values()) + tuple(frame.forward.kernels.values())
            packed = ExecutionPlan(device, list(frame.forward.plan)*4, labeled=True)
            with packed.capture(resources=resources) as fused_replays:
                ordinary = measure_cuda(device, frame.forward.run,
                                        warmup=2, samples=5, repeats=4)
                joined = measure_cuda(device, fused_replays.launch,
                                      warmup=1, samples=5, repeats=1)
            actual = frame.forward.buffers[frame.prediction].to_numpy().reshape(frame.rows, chunk)
            if actual[0, :len(args[1][0])].tolist() != result[0]:
                raise ValueError('Repeated verifier changed predictions')
            item = dict(chunk=chunk, live_tokens=len(args[1][0]),
                        position=int(target.positions[args[0][0]]),
                        operation_count=len(frame.forward.plan),
                        existing_graph=ordinary, four_replays_in_one_graph=joined,
                        joined_per_forward_gpu_ms=joined['median_gpu_seconds']*250,
                        ordinary_per_forward_gpu_ms=ordinary['median_gpu_seconds']*1000,
                        predictions_exact=True)
            micro_rows.append(item)
            micro_done.add(chunk)
            print('VERIFIER_REPLAY', json.dumps(item), flush=True)
            return result

        for item in workload['cases']:
            case_id = item['id']
            print('MEASURE', case_id, flush=True)
            ids = item['token_ids']
            with SpeculativeSession(decoder, batch=1, chunk=512, timeout=120) as session:
                row = run(session, Case(1, len(ids), 128), prompts=[ids], warmup=1, repeats=3)
                expected = list(session.requests[0].output)
                row['speculation'] = session.stats
            if [expected] != previous[case_id]['output_ids']:
                raise ValueError('Unchanged short output differs from previous Q8 measurement')
            row.update(text_case={k:v for k,v in item.items() if k!='token_ids'},
                       output_ids=[expected], previous_output_exact=True)
            phase_samples.clear()
            originals = {'propose':head.propose, 'verify':target.verify,
                         'commit':target.commit, 'repair':head.repair}
            Forward.run = instrument_forward
            for obj, method in ((head,'propose'), (target,'verify'),
                                (target,'commit'), (head,'repair')):
                setattr(obj, method, make_phase(originals[method],
                                               'proposal' if method=='propose' else method))
            mode = 'phases'
            try:
                with SpeculativeSession(decoder, batch=1, chunk=512, timeout=120) as session:
                    session.reset()
                    session.prefill([ids])
                    device.synchronize()
                    start = time.perf_counter()
                    session.decode(127)
                    device.synchronize()
                    total_ms = (time.perf_counter()-start)*1000
                    if session.requests[0].output != expected:
                        raise ValueError('Phase instrumentation changed output')
                    row['diagnostic_speculation'] = session.stats
                phases = {}
                for name, samples in phase_samples.items():
                    phases[name] = dict(
                        calls=len(samples), total_wall_ms=sum(s['wall_ms'] for s in samples),
                        total_forward_gpu_span_ms=sum(f['gpu_span_ms'] for s in samples for f in s['forwards']),
                        median_wall_ms=statistics.median(s['wall_ms'] for s in samples),
                        median_forward_gpu_span_ms=statistics.median(
                            sum(f['gpu_span_ms'] for f in s['forwards']) for s in samples),
                        samples=samples)
                row['phase_attribution'] = dict(
                    total_ms=total_ms, phases=phases,
                    outside_phase_wall_ms=total_ms-sum(p['total_wall_ms'] for p in phases.values()),
                    verification_shapes=dict(Counter(s['chunk'] for s in phase_samples['verify'])),
                    output_exact=True)
            finally:
                mode = 'none'
                Forward.run = original_forward
                for obj, method in ((head,'propose'), (target,'verify'),
                                    (target,'commit'), (head,'repair')):
                    setattr(obj, method, originals[method])
            # Separate run: repeating the verifier must not contaminate phase
            # timings. Exact end-to-end output checks recurrent state as well.
            mode = 'micro'
            target.verify = verifier_micro
            try:
                with SpeculativeSession(decoder, batch=1, chunk=512, timeout=120) as session:
                    session.reset()
                    session.prefill([ids])
                    session.decode(127)
                    if session.requests[0].output != expected:
                        raise ValueError('Fixed-input verifier replay changed generation')
            finally:
                target.verify = originals['verify']
                mode = 'none'
            row['correctness'] = dict(correct=True, scope=metadata['qualification'])
            report['results'].append(row)
            metadata['verifier_replay'] = micro_rows
            rec.update(report)
            write(OUT/'report.json', report)
            print('RESULT', case_id, json.dumps(dict(
                decode_tps=row['summary']['decode_tokens_per_second']['median'],
                phases={name:{k:v for k,v in data.items() if k!='samples'}
                        for name,data in row['phase_attribution']['phases'].items()},
                shapes=row['phase_attribution']['verification_shapes'],
                outside_ms=row['phase_attribution']['outside_phase_wall_ms'])), flush=True)
        totals = defaultdict(float)
        for row in report['results']:
            attribution = row['phase_attribution']
            totals['round_wall_ms'] += attribution['total_ms']
            totals['outside_phase_wall_ms'] += attribution['outside_phase_wall_ms']
            for phase, values in attribution['phases'].items():
                totals[phase+'_wall_ms'] += values['total_wall_ms']
                totals[phase+'_forward_gpu_span_ms'] += values['total_forward_gpu_span_ms']
        metadata['aggregate'] = dict(totals)
        passed = True
except BaseException as error:
    metadata['error'] = f'{type(error).__name__}: {error}'
    metadata['traceback'] = traceback.format_exc()
    print(metadata['traceback'], flush=True)
finally:
    Forward.run = original_forward
    metadata['cpu_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    rec.artifact(__file__, 'profiling/runner.py')
    rec.artifact(WORKLOAD_PATH, 'benchmark/text-workload.json')
    rec.program(TARGET)
    rec.program(DRAFT)
    write(OUT/'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

"""Whole-model original/parallel/original decode over identical GPU snapshots."""
import json
from contextlib import ExitStack
from pathlib import Path
import statistics
import time
import traceback
import numpy as np
from akas.artifacts.read import read_program
from akas.benchmarking.history import Recorder, write
from akas.benchmarking.session import Session
from akas.execution.plan import ExecutionPlan, call_bindings
from akas.graph import Graph
from akas.models.lfm2.model import LFM2MoE
from akas.optimization.evaluation import prompts
from akas.optimization.planning import estimate
from wgpu_profile import timestamp_device

program = Path('build/lfm-moe/manual-v2-matched/webgpu/program')
root = Path('build/lfm-moe/argmax-model-webgpu')
root.mkdir(exist_ok=True)
reference = json.loads((program.parent / 'report.json').read_text())
needed = max(estimate(Graph.from_metadata(json.loads((program / p['path']).read_text())['forward']))['largest_buffer_bytes']
             for p in read_program(program)['phases'])
proof = json.loads(Path('build/lfm-moe/parallel-argmax/report.json').read_text())
assert all(c['exact'] for c in proof['metadata']['checks'])
meta = dict(reference['metadata'])
meta.update(purpose='kernel', protocol='snapshot-decode-v1', max_buffer_size=needed,
    kernel_evidence_run_id='20261020191837105510-7798741d',
    qualification='Whole-model original/parallel/original native decode on identical GPU state snapshots. Exact 128-token IDs versus the retained F16 WebGPU reference; unchanged weights, routing, operands, accumulators and KV precision. 1 warmup and 3 timed 127-step repeats. Decode-only research overlay, no prefill or HTTP performance claim.')
rec = Recorder('benchmarks', meta, kind='algorithm',
    change='Validate exact hierarchical vocabulary argmax under full-model streaming weights and real request state')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False
try:
    with timestamp_device(max_buffer_size=needed) as device:
        with LFM2MoE(meta['checkpoint'], program, device, graphs=True) as model, ExitStack() as stack:
            modules = [device.load(Path('build/lfm-moe/parallel-argmax') / f'1-128000-1024-{name}.tbin') for name in ('partial', 'merge')]
            for module in modules:
                stack.callback(module.release)
            values, indices = device.empty((1, 125), 'float32'), device.empty((1, 125), 'int32')
            stack.callback(values.release)
            stack.callback(indices.release)
            forward = model.batches.select(1, 1).forward
            original = forward.graph
            plan = ExecutionPlan(device, labeled=True)
            replacements = 0
            for entry, call in zip(forward.plan, forward.definition.calls):
                if call.kind == 'argmax':
                    assert (call.parameters['r'], call.parameters['c']) == (1, 128000)
                    b = call_bindings(entry[0], entry[1])
                    plan.bind(modules[0], (b['x'], values, indices), label=entry[2])
                    plan.bind(modules[1], (values, indices, b['enabled'], b['out']), label=entry[2])
                    replacements += 1
                else:
                    plan.append(entry)
            assert replacements == 1
            candidate = plan.capture(resources=tuple(forward.buffers.values()) + tuple(forward.kernels.values()) + tuple(modules) + (values, indices))
            stack.callback(candidate.close)
            stack.callback(lambda: setattr(forward, 'graph', original))
            print('OVERLAY_READY', replacements, flush=True)
            for prefix in (32, 8192):
                print('PREFILL_SNAPSHOT', prefix, flush=True)
                with Session(model, batch=1, chunk=512, timeout=120) as session, ExitStack() as snapshots:
                    forward.graph = original
                    session.reset()
                    first = list(session.prefill(prompts(model.config.vocab, 1, prefix)))
                    positions = model.positions.copy()
                    states = {}
                    for name, buffer in model.batches.states.items():
                        clone = device.empty(buffer.shape, str(buffer.dtype))
                        clone.copy_from(buffer)
                        states[name] = clone
                        snapshots.callback(clone.release)
                    device.synchronize()

                    def restore():
                        for name, clone in states.items():
                            model.batches.states[name].copy_from(clone)
                        model.positions[:] = positions
                        session.tokens = list(first)
                        session.feedback_ready = False
                        session.deadline = time.perf_counter() + 120
                        device.synchronize()

                    expected = next(r for r in reference['results'] if r['case']['prompt'] == prefix)['output_ids'][0]
                    for label in ('original-before', 'parallel', 'original-after'):
                        forward.graph = candidate if label == 'parallel' else original
                        restore()
                        ids = list(first)
                        for _ in range(127):
                            ids.extend(session.decode(1))
                        if ids != expected:
                            raise ValueError(f'{label} changes qualified model IDs at {prefix}')
                        samples = []
                        for iteration in range(4):
                            restore()
                            start = time.perf_counter()
                            session.decode(127, read=False)
                            device.synchronize()
                            elapsed = time.perf_counter() - start
                            if iteration:
                                samples.append(127 / elapsed)
                        result = dict(prefix=prefix, variant=label, decode_tokens_per_second=statistics.median(samples),
                                      samples=samples, identical_output=True)
                        meta.setdefault('decode_comparisons', []).append(result)
                        write(root / 'report.json', report)
                        print('RESULT', json.dumps(result), flush=True)
    passed = True
except BaseException:
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    rec.program(program)
    for path in (__file__, 'build/lfm-moe/parallel_argmax.py', 'build/lfm-moe/compile-parallel-argmax.py', 'build/lfm-moe/wgpu_profile.py'):
        rec.artifact(path, 'kernels/' + Path(path).name)
    for name in ('partial', 'merge'):
        rec.artifact(Path('build/lfm-moe/parallel-argmax') / f'1-128000-1024-{name}.tbin', 'overlay/' + name + '.tbin')
    rec.artifact(program.parent / 'report.json', 'reference/webgpu-report.json')
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

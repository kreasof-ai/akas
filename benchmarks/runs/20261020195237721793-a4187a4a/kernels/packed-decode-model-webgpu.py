"""Whole-model measured decode candidates over identical GPU snapshots."""
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
root = Path('build/lfm-moe/packed-decode-model-webgpu')
root.mkdir(exist_ok=True)
reference = json.loads((program.parent / 'report.json').read_text())
needed = max(estimate(Graph.from_metadata(json.loads((program / p['path']).read_text())['forward']))['largest_buffer_bytes']
             for p in read_program(program)['phases'])
proof = json.loads(Path('build/lfm-moe/parallel-argmax/report.json').read_text())
assert all(c['exact'] for c in proof['metadata']['checks'])
meta = dict(reference['metadata'])
meta.update(purpose='kernel', protocol='snapshot-decode-v1', max_buffer_size=needed,
    kernel_evidence_run_ids=['20261020191837105510-7798741d','20261020192849707213-cd589c64','20261020193348533055-bc9b08e1','20261020195031655272-927c4e8a'],
    qualification='Whole-model original/candidates/original native decode on identical GPU state snapshots. Every candidate must match all 128 IDs versus the retained F16 WebGPU reference. No operand/accumulator/KV precision reduction; faster router threads and packed expert lanes regroup F32 additions and have independent F64 product checks; original F32 coefficients are preserved. 1 warmup and 3 timed 127-step repeats. Decode-only overlays, no prefill or HTTP performance claim.')
rec = Recorder('benchmarks', meta, kind='algorithm',
    change='Compare packed F32 expert products with the qualified scalar-decode incumbent under identical full-model request states')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False
try:
    with timestamp_device(max_buffer_size=needed) as device:
        with LFM2MoE(meta['checkpoint'], program, device, graphs=True) as model, ExitStack() as stack:
            paths = dict(arg_partial=Path('build/lfm-moe/parallel-argmax/1-128000-1024-partial.tbin'),
                         arg_merge=Path('build/lfm-moe/parallel-argmax/1-128000-1024-merge.tbin'),
                         router_exact=Path('build/lfm-moe/router-projection/full-1-32.tbin'),
                         router_fast=Path('build/lfm-moe/router-projection/full-1-256.tbin'),
                         gate=Path('build/lfm-moe/webgpu-packed-expert-decode/gate-1-1.tbin'),
                         down=Path('build/lfm-moe/webgpu-packed-expert-decode/down-1-1.tbin'),
                         scalar_gate=Path('build/lfm-moe/webgpu-expert-decode/gate-1-4.tbin'),
                         scalar_down=Path('build/lfm-moe/webgpu-expert-decode/down-1-4.tbin'))
            modules = {name: device.load(path) for name, path in paths.items()}
            for module in modules.values():
                stack.callback(module.release)
            values, indices = device.empty((1, 125), 'float32'), device.empty((1, 125), 'int32')
            stack.callback(values.release)
            stack.callback(indices.release)
            forward = model.batches.select(1, 1).forward
            original = forward.graph
            candidates = {}
            for name, router, experts in (('incumbent', 'router_fast', 'scalar'),
                                           ('packed-exact', 'router_exact', 'packed'),
                                           ('packed-fast', 'router_fast', 'packed')):
                plan = ExecutionPlan(device, labeled=True)
                counts = dict(argmax=0, router=0, experts=0)
                for entry, call in zip(forward.plan, forward.definition.calls):
                    p = call.parameters
                    if call.kind == 'argmax':
                        assert (p['r'], p['c']) == (1, 128000)
                        b = call_bindings(entry[0], entry[1])
                        plan.bind(modules['arg_partial'], (b['x'], values, indices), label=entry[2])
                        plan.bind(modules['arg_merge'], (values, indices, b['enabled'], b['out']), label=entry[2])
                        counts['argmax'] += 1
                    elif call.kind == 'linear' and not p.get('ggml', False) and (p['r'], p['k'], p['o']) == (1, 2048, 64):
                        b = call_bindings(entry[0], entry[1])
                        module = modules[router]
                        plan.bind(module, tuple(b[a['name']] for a in module.manifest['arguments']), label=entry[2])
                        counts['router'] += 1
                    elif experts and call.kind == 'quantized_experts':
                        b = call_bindings(entry[0], entry[1])
                        module = modules[('scalar_down' if p.get('routed_input') else 'scalar_gate') if experts == 'scalar' else ('down' if p.get('routed_input') else 'gate')]
                        plan.bind(module, tuple(b[a['name']] for a in module.manifest['arguments']), label=entry[2])
                        counts['experts'] += 1
                    else:
                        plan.append(entry)
                assert counts == dict(argmax=1, router=38, experts=114 if experts else 0)
                graph = plan.capture(resources=tuple(forward.buffers.values()) + tuple(forward.kernels.values()) + tuple(modules.values()) + (values, indices))
                stack.callback(graph.close)
                candidates[name] = graph
                print('OVERLAY_READY', name, counts, flush=True)
            stack.callback(lambda: setattr(forward, 'graph', original))
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
                    for label in ('original-before', *candidates, 'original-after'):
                        forward.graph = candidates.get(label, original)
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
    for name, path in paths.items():
        rec.artifact(path, 'overlay/' + name + '.tbin')
    rec.artifact('build/lfm-moe/router_projection.py', 'kernels/router_projection.py')
    rec.artifact('build/lfm-moe/wgpu_expert_decode.py', 'kernels/wgpu_expert_decode.py')
    rec.artifact('build/lfm-moe/wgpu_packed_experts.py', 'kernels/wgpu_packed_experts.py')
    rec.artifact('build/lfm-moe/wgpu_q4_dot.py', 'kernels/wgpu_q4_dot.py')
    rec.artifact(program.parent / 'report.json', 'reference/webgpu-report.json')
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

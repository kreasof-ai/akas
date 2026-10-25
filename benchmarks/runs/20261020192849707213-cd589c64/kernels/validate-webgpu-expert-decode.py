"""Independent bounded CPU references and warm/cold full-size expert trials."""
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import traceback
from types import SimpleNamespace
import numpy as np
from akas.artifacts.read import artifact_path, read_program
from akas.benchmarking.history import Recorder, write
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.quantization.weights.ggml.validation import sample_weight, decode_rows
from wgpu_profile import Profile, timestamp_device

root = Path('build/lfm-moe/webgpu-expert-decode')
program = Path('build/lfm-moe/manual-v2-matched/webgpu/program')
index = read_program(program)
manifest = json.loads((program / next(p['path'] for p in index['phases'] if p['phase'] == 'decode')).read_text())
meta = dict(runtime='akas', purpose='kernel', protocol='ordered-webgpu-pass-profile-v1', provider='webgpu',
    target='webgpu-portable-v1', device=dict(name='NVIDIA L4', provider='webgpu'), weight_format='q4_0',
    operand_dtype='float32', accumulator_dtype='float32', shape_timeout_seconds=120,
    qualification='Independent bounded CPU F64 products with original F32 operands; strict bitwise comparison against retained AOT. Distinct expert scales, odd output tails, inactive experts, immutable input bytes. Warm GPU pass timing and cold-cache proxy with a preceding 128 MiB write excluded from kernel interval. No model/serving throughput claim.')
rec = Recorder('benchmarks', meta, kind='algorithm',
    change='Test WebGPU expert loop unrolling and activation reuse with unchanged F32 lane addition order')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False

def timer(device, calls):
    plan = ExecutionPlan(device, labeled=True)
    for module, args in calls:
        plan.bind(module, args, label=None)
    return Profile(SimpleNamespace(device=device, plan=plan,
        definition=SimpleNamespace(calls=[SimpleNamespace(kind='trial', parameters={}) for _ in calls])))

def reference(arrays, p):
    out = np.zeros_like(arrays['out'])
    for rank, expert in enumerate(arrays['ids'][0]):
        if not arrays['counts'][expert]:
            continue
        x = arrays['x'][0, rank] if p['routed_input'] else arrays['x'][0]
        for start in range(0, p['o'], 128):
            stop = min(p['o'], start + 128)
            w = decode_rows(arrays['w'], 2, p['k'], expert * p['o'] + start, expert * p['o'] + stop)
            out[0, rank, start:stop] = w.astype('float64') @ x.astype('float64')
    return out

try:
    with timestamp_device() as device, ExitStack() as owned:
        flush = device.load(root / 'flush.tbin')
        owned.callback(flush.release)
        cache = device.empty((33554432,), 'uint32')
        owned.callback(cache.release)
        for kind, depth, columns, experts in (('small-gate', 256, 37, 6), ('small-down', 256, 37, 6),
                                            ('gate', 2048, 1536, 64), ('down', 1536, 2048, 64)):
            p = dict(r=1, top=4, experts=experts, k=depth, o=columns, input_dtype='float32', round_inputs=False,
                     routed_input=kind.endswith('down'), weight_parts=({'rows': experts * columns, 'format': 'ggml', 'options': {'encoding': 2}},))
            baseline_path = root / f'{kind}-original-1.tbin' if experts == 6 else artifact_path(program,
                next(k for k in manifest['kernels'].values() if k['kind'] == 'quantized_experts' and k['parameters']['k'] == depth))
            with ExitStack() as group:
                baseline = device.load(baseline_path)
                group.callback(baseline.release)
                arrays = sample_arguments(baseline.manifest['arguments'])
                sample_weight(arrays['w'], 2, decode=False)
                blocks = arrays['w'].view('uint8').reshape(experts, columns, depth // 32, 18)
                scales = blocks[..., :2].view('float16')[..., 0]
                for expert in range(experts):
                    scales[expert] *= np.float16(.5 + expert / max(1, experts - 1))
                arrays['ids'][0] = [0, 1, 3, 5] if experts == 6 else [2, 17, 39, 63]
                arrays['counts'].fill(0)
                arrays['counts'][arrays['ids'][0]] = 1
                arrays['out'].fill(1.25)
                expected = reference(arrays, p)
                buffers = {name: device.from_numpy(value) for name, value in arrays.items()}
                for buffer in buffers.values():
                    group.callback(buffer.release)
                baseline.launch(*(buffers[a['name']] for a in baseline.manifest['arguments']))
                original = buffers['out'].to_numpy()
                np.testing.assert_allclose(original, expected, rtol=.002, atol=.0002)
                options = [(2, 8)] if experts == 6 else [(1, 1), (1, 4), (1, 8), (2, 4), (2, 8), (4, 4)]
                candidates = [('original', baseline)]
                for width, unroll in options:
                    module = device.load(root / f'{kind}-{width}-{unroll}.tbin')
                    group.callback(module.release)
                    candidates.append((f'{width}-{unroll}', module))
                for geometry, module in candidates:
                    args = tuple(buffers[a['name']] for a in module.manifest['arguments'])
                    device.write(buffers['out'], arrays['out'])
                    module.launch(*args)
                    actual = buffers['out'].to_numpy()
                    np.testing.assert_allclose(actual, expected, rtol=.002, atol=.0002)
                    np.testing.assert_array_equal(actual.view('uint32'), original.view('uint32'))
                    for name in arrays:
                        if name != 'out':
                            np.testing.assert_array_equal(buffers[name].to_numpy(), arrays[name])
                    inactive = arrays['counts'].copy()
                    inactive[arrays['ids'][0, -1]] = 0
                    device.write(buffers['counts'], inactive)
                    module.launch(*args)
                    np.testing.assert_array_equal(buffers['out'].to_numpy()[0, -1], np.zeros(columns, dtype='float32'))
                    device.write(buffers['counts'], np.zeros_like(inactive))
                    module.launch(*args)
                    np.testing.assert_array_equal(buffers['out'].to_numpy(), np.zeros_like(original))
                    device.write(buffers['counts'], arrays['counts'])
                    check = dict(kind=kind, geometry=geometry, parameters=p, cpu_reference=True,
                                 bitwise=True, immutable_inputs=True, inactive_zero=True)
                    if experts == 64:
                        for cold in (False, True):
                            calls = [(flush, (cache,)), (module, args)] if cold else [(module, args)]
                            profile = timer(device, calls)
                            try:
                                times = []
                                for iteration in range(15):
                                    profile.launch()
                                    values = profile.read()
                                    if iteration >= 3:
                                        times.append(values[-1])
                            finally:
                                profile.close()
                            check['cold' if cold else 'warm'] = dict(gpu_ms=statistics.median(times), samples_ms=times)
                    meta.setdefault('checks', []).append(check)
                    print('PASS', kind, geometry, json.dumps({k: check[k] for k in ('warm', 'cold') if k in check}), flush=True)
                    write(root / 'report.json', report)
    passed = True
except BaseException:
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    for path in (__file__, 'build/lfm-moe/wgpu_expert_decode.py', 'build/lfm-moe/compile-webgpu-expert-decode.py', 'build/lfm-moe/wgpu_profile.py'):
        rec.artifact(path, 'kernels/' + Path(path).name)
    for path in sorted(root.glob('*.tbin')):
        rec.artifact(path, 'kernels/' + path.name)
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

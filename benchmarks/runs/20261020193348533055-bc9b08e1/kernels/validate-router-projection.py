"""Check F32 router projection geometry against independent products and AOT."""
from contextlib import ExitStack
import json
from pathlib import Path
import statistics
import traceback
from types import SimpleNamespace
import numpy as np
from akas.artifacts.read import read_program, artifact_path
from akas.benchmarking.history import Recorder, write
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from wgpu_profile import Profile, timestamp_device

root = Path('build/lfm-moe/router-projection')
program = Path('build/lfm-moe/manual-v2-matched/webgpu/program')
top = read_program(program)
manifest = json.loads((program / next(p['path'] for p in top['phases'] if p['phase'] == 'decode')).read_text())
old = artifact_path(program, next(k for k in manifest['kernels'].values() if k['kind'] == 'linear' and k['parameters']['o'] == 64))
meta = dict(runtime='akas', purpose='kernel', protocol='ordered-webgpu-pass-profile-v1', provider='webgpu',
    target='webgpu-portable-v1', device=dict(name='NVIDIA L4', provider='webgpu'),
    operand_dtype='float32', accumulator_dtype='float32',
    qualification='Independent CPU F64 products, strided outputs and non-multiple tails; unchanged input bytes. Output grouping preserves per-lane sums; different thread counts alter F32 summation geometry. Warm and cold-cache proxy GPU timings bracketed by AOT before/after; no serving claim.')
rec = Recorder('benchmarks', meta, kind='algorithm', change='Expose small router projections to more workgroups without reducing operand or accumulator precision')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False

def capture(device, calls):
    plan = ExecutionPlan(device, labeled=True)
    for module, args in calls:
        plan.bind(module, args, label=None)
    return Profile(SimpleNamespace(device=device, plan=plan,
        definition=SimpleNamespace(calls=[SimpleNamespace(kind='trial', parameters={}) for _ in calls])))

try:
    with timestamp_device() as device, ExitStack() as owned:
        flush = device.load('build/lfm-moe/webgpu-expert-decode/flush.tbin')
        owned.callback(flush.release)
        cache = device.empty((33554432,), 'uint32')
        owned.callback(cache.release)
        for kind, p in (('small', dict(r=3, k=257, o=7, output_stride=15, output_offset=3, matrix_output=True)),
                        ('full', dict(r=1, k=2048, o=64))):
            with ExitStack() as group:
                first = device.load(root / 'small-1-128.tbin' if kind == 'small' else old)
                group.callback(first.release)
                arrays = sample_arguments(first.manifest['arguments'])
                expected = arrays['out'].copy()
                stride, offset = p.get('output_stride', p['o']), p.get('output_offset', 0)
                expected.reshape(p['r'], stride)[:, offset:offset + p['o']] = arrays['x'].reshape(p['r'], p['k']).astype('float64') @ arrays['w'].reshape(p['o'], p['k']).astype('float64').T
                buffers = {name: device.from_numpy(value) for name, value in arrays.items()}
                for buffer in buffers.values():
                    group.callback(buffer.release)
                args = tuple(buffers[a['name']] for a in first.manifest['arguments'])
                first.launch(*args)
                original = buffers['out'].to_numpy()
                np.testing.assert_allclose(original, expected, rtol=2e-5, atol=2e-6)
                choices = [('original-before', first)]
                if kind == 'full':
                    for columns, threads in ((4, 32), (1, 32), (1, 64), (1, 128), (1, 256), (2, 128), (2, 256)):
                        module = device.load(root / f'full-{columns}-{threads}.tbin')
                        group.callback(module.release)
                        choices.append((f'{columns}-{threads}', module))
                    choices.append(('original-after', first))
                for geometry, module in choices:
                    args = tuple(buffers[a['name']] for a in module.manifest['arguments'])
                    device.write(buffers['out'], arrays['out'])
                    module.launch(*args)
                    actual = buffers['out'].to_numpy()
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
                    for name in arrays:
                        if name != 'out':
                            np.testing.assert_array_equal(buffers[name].to_numpy(), arrays[name])
                    if kind == 'small':
                        mask = np.ones_like(actual, dtype=bool)
                        mask[:, offset:offset + p['o']] = False
                        np.testing.assert_array_equal(actual[mask], arrays['out'][mask])
                    check = dict(kind=kind, geometry=geometry, parameters=p, cpu_reference=True,
                        immutable_inputs=True, bitwise=bool(np.array_equal(actual.view('uint32'), original.view('uint32'))),
                        maximum_absolute_error=float(np.max(np.abs(actual.astype('float64') - expected))))
                    if kind == 'full':
                        for cold in (False, True):
                            profile = capture(device, [(flush, (cache,)), (module, args)] if cold else [(module, args)])
                            try:
                                samples = []
                                for iteration in range(15):
                                    profile.launch()
                                    timing = profile.read()
                                    if iteration >= 3:
                                        samples.append(timing[-1])
                            finally:
                                profile.close()
                            check['cold' if cold else 'warm'] = dict(gpu_ms=statistics.median(samples), samples_ms=samples)
                    meta.setdefault('checks', []).append(check)
                    print('PASS', geometry, check.get('cold'), 'bitwise', check['bitwise'], flush=True)
                    write(root / 'report.json', report)
    passed = True
except BaseException:
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    for path in (__file__, 'build/lfm-moe/router_projection.py', 'build/lfm-moe/compile-router-projection.py', 'build/lfm-moe/wgpu_profile.py'):
        rec.artifact(path, 'kernels/' + Path(path).name)
    for path in sorted(root.glob('*.tbin')):
        rec.artifact(path, 'kernels/' + path.name)
    rec.artifact(old, 'reference/router.tbin')
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

"""Independent exact edge cases and ordered GPU timing for vocabulary reduction."""
import json
from pathlib import Path
import statistics
import traceback
from types import SimpleNamespace
import numpy as np
from akas.artifacts.read import read_program, artifact_path
from akas.benchmarking.history import Recorder, write
from akas.execution.plan import ExecutionPlan
from wgpu_profile import Profile, timestamp_device

root = Path('build/lfm-moe/parallel-argmax')
program = Path('build/lfm-moe/manual-v2-matched/webgpu/program')
top = read_program(program)
manifest = json.loads((program / next(p['path'] for p in top['phases'] if p['phase'] == 'decode')).read_text())
old = artifact_path(program, next(k for k in manifest['kernels'].values() if k['kind'] == 'argmax'))
meta = dict(runtime='akas', purpose='kernel', protocol='ordered-webgpu-pass-profile-v1', provider='webgpu',
    target='webgpu-portable-v1', device=dict(name='NVIDIA L4', provider='webgpu'),
    qualification='Independent NumPy first-maximum oracle; cross-tile ties, non-multiple tails, infinities, ignored NaNs, all-invalid sentinel and inactive output preservation. Diagnostic GPU pass times include both new dispatches; no serving claim.')
rec = Recorder('benchmarks', meta, kind='algorithm', change='Parallelize exact vocabulary argmax across input blocks and merge first-index winners')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False

def prepare(device, calls, kinds):
    plan = ExecutionPlan(device, labeled=True)
    for module, args in calls:
        plan.bind(module, args, label=None)
    return Profile(SimpleNamespace(device=device, plan=plan,
        definition=SimpleNamespace(calls=[SimpleNamespace(kind=k, parameters={}) for k in kinds])))

def expected(values, enabled):
    result = np.full(len(enabled), 99, dtype='int32')
    for row in range(len(enabled)):
        if not enabled[row]:
            continue
        valid = np.where(np.isnan(values[row]), -np.inf, values[row])
        peak = valid.max()
        result[row] = values.shape[1] if peak == -np.inf else int(np.flatnonzero(valid == peak)[0])
    return result

try:
    with timestamp_device() as device:
        for rows, columns, block in ((1, 128000, 1024), (2, 2051, 512)):
            parts = (columns + block - 1) // block
            modules = [device.load(root / f'{rows}-{columns}-{block}-{name}.tbin') for name in ('partial', 'merge')]
            buffers = [device.empty((rows * columns,), 'float32'), device.empty((rows, parts), 'float32'),
                device.empty((rows, parts), 'int32'), device.empty((rows,), 'int32'), device.empty((rows,), 'int32')]
            x, values, indices, enabled, out = buffers
            timer = prepare(device, [(modules[0], (x, values, indices)), (modules[1], (values, indices, enabled, out))], ['argmax_partial', 'argmax_merge'])
            try:
                rng = np.random.default_rng(91)
                checks = []
                for case in ('normal', 'tie', 'nan', 'negative-infinity', 'positive-infinity', 'all-nan', 'signed-zero', 'inactive'):
                    data = rng.normal(size=(rows, columns)).astype('float32')
                    mask = np.ones(rows, dtype='int32')
                    if case == 'tie':
                        data[:, [5, block + 3, columns - 1]] = 10
                        if rows > 1:
                            mask[1] = 0
                    elif case == 'nan':
                        data[:, ::7] = np.nan
                        data[:, 31] = 10
                    elif case == 'negative-infinity':
                        data.fill(-np.inf)
                    elif case == 'positive-infinity':
                        data[:, [31, columns - 1]] = np.inf
                    elif case == 'all-nan':
                        data.fill(np.nan)
                    elif case == 'signed-zero':
                        data.fill(-1)
                        data[:, 5] = 0
                        data[:, columns - 1] = -0.0
                    elif case == 'inactive':
                        mask.fill(0)
                    device.write(x, data.reshape(-1))
                    device.write(enabled, mask)
                    device.write(out, np.full(rows, 99, dtype='int32'))
                    timer.launch()
                    np.testing.assert_array_equal(out.to_numpy(), expected(data, mask))
                    np.testing.assert_array_equal(x.to_numpy(), data.reshape(-1))
                    np.testing.assert_array_equal(enabled.to_numpy(), mask)
                    checks.append(case)
                check = dict(rows=rows, columns=columns, block=block, cases=checks, exact=True)
                if rows == 1:
                    device.write(x, rng.normal(size=rows * columns).astype('float32'))
                    device.write(enabled, np.ones(rows, dtype='int32'))
                    candidate = []
                    for iteration in range(15):
                        timer.launch()
                        timing = timer.read()
                        if iteration >= 3:
                            candidate.append(sum(timing))
                    with device.load(old) as baseline:
                        before = prepare(device, [(baseline, (x, out, enabled))], ['argmax'])
                        try:
                            measured = []
                            for iteration in range(15):
                                before.launch()
                                timing = before.read()
                                if iteration >= 3:
                                    measured.append(sum(timing))
                        finally:
                            before.close()
                    check.update(baseline_gpu_ms=statistics.median(measured), candidate_gpu_ms=statistics.median(candidate),
                                 baseline_samples_ms=measured, candidate_samples_ms=candidate)
                meta.setdefault('checks', []).append(check)
                print('PASS', json.dumps(check), flush=True)
            finally:
                timer.close()
                for module in modules:
                    module.release()
                for buffer in buffers:
                    buffer.release()
    passed = True
except BaseException:
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    for path in (__file__, 'build/lfm-moe/parallel_argmax.py', 'build/lfm-moe/compile-parallel-argmax.py', 'build/lfm-moe/wgpu_profile.py'):
        rec.artifact(path, 'kernels/' + Path(path).name)
    for path in sorted(root.glob('*.tbin')):
        rec.artifact(path, 'kernels/' + path.name)
    rec.artifact(old, 'reference/argmax.tbin')
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

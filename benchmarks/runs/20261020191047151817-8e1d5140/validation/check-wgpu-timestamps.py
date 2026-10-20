"""Exercise the diagnostic timer with a real, independently checked operation."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from akas.artifacts.read import read_program, artifact_path
from akas.execution.plan import ExecutionPlan
from wgpu_profile import Profile, timestamp_device

program = Path('build/lfm-moe/manual-v2-matched/webgpu/program')
top = read_program(program)
phase = next(p for p in top['phases'] if p['phase'] == 'prefill' and p['chunk'] == 32)
manifest = json.loads((program / phase['path']).read_text())
entry = next(k for k in manifest['kernels'].values() if k['kind'] == 'add')
with timestamp_device() as device:
    with device.load(artifact_path(program, entry)) as module:
        x = np.linspace(-1, 1, 65536, dtype='float32')
        y = np.full_like(x, 0.25)
        buffers = dict(x=device.from_numpy(x), y=device.from_numpy(y), out=device.from_numpy(np.zeros_like(x)))
        try:
            plan = ExecutionPlan(device, labeled=True)
            plan.bind(module, tuple(buffers[a['name']] for a in module.manifest['arguments']), label='add')
            forward = SimpleNamespace(device=device, plan=plan,
                definition=SimpleNamespace(calls=[SimpleNamespace(kind='add', parameters=entry['parameters'])]))
            profile = Profile(forward)
            try:
                samples = []
                for _ in range(3):
                    profile.launch()
                    samples.extend(profile.read())
                np.testing.assert_array_equal(buffers['out'].to_numpy(), x + y)
                np.testing.assert_array_equal(buffers['x'].to_numpy(), x)
                np.testing.assert_array_equal(buffers['y'].to_numpy(), y)
                assert all(0 < value < 1000 for value in samples)
                print(json.dumps(dict(correct=True, immutable_inputs=True, samples_ms=samples)), flush=True)
            finally:
                profile.close()
        finally:
            for buffer in buffers.values():
                buffer.release()

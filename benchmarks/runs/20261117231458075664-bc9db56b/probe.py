import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import ExitStack
import traceback

import numpy as np
import tensor
from akas.benchmarking.history import Recorder
from akas.kernels.contracts import sample_arguments
from akas.kernels.definition import KernelSpec
from akas.ops.vocabulary.validation import validation as vocabulary_validation
from akas.ops.attention.validation import canonical_validation

base = '401c34a8013439a5ed95e03b7538a4fb85f51f8d'
root = Path(os.environ['PARITY_ROOT'])
metadata = dict(purpose='kernel-qualification', protocol='prior-kernel-bitwise-v1', provider='cuda', target='sm_89', predecessor=base, checks=[])
rec = Recorder('benchmarks', metadata, kind='qualification', change='Replay prior default vocabulary and floating attention CUDA kernels against the current arithmetic')
rec.artifact(__file__, 'probe.py')
cases = {
    ('akas.ops.vocabulary.kernel', 'argmax', 'argmax'): [dict(r=5,c=c) for c in (64, 513, 8197, 40980)],
    ('akas.ops.vocabulary.kernel', 'embedding', 'embedding'): [dict(r=r,c=c,v=43,start=16) for r,c in ((1,64),(7,192),(64,257),(128,1024))],
    ('akas.ops.attention.cuda', 'floating', 'attention'): [dict(r=2*chunk,h=4,kh=2,d=64,capacity=2*chunk,chunk=chunk,kv_dtype='float16',unit_scales=True) for chunk in (16,32,64)],
}
try:
    with tensor.Device() as device:
        metadata['device'] = dict(device.info)
        for (module, factory, kind), geometries in cases.items():
            legacy = 'historical_' + module.replace('.', '_')
            historical = root / (legacy + '.py')
            path = 'src/' + module.replace('.', '/') + '.py'
            historical.write_bytes(subprocess.check_output(['git','show',f'{base}:{path}']))
            rec.artifact(historical, f'predecessor/{path}')
            historical.write_text(f'__package__ = {module.rsplit(".", 1)[0]!r}\n' + historical.read_text())
            for i, p in enumerate(geometries):
                sigs, outcomes, kernels = [], [], []
                kwargs = dict(kind=kind, parameters={**p,'provider':'cuda'})
                for tag, mod in [('old', legacy), ('new', module)]:
                    source = root / f'{factory}-{i}-{tag}.py'
                    source.write_text(f'from akas.kernels.definition import KernelSpec\nfrom {mod} import {factory}\n\ndef tensor_export():\n    return {{"kernel": {factory}(KernelSpec(**{kwargs!r}), {{}})}}\n')
                    binary = source.with_suffix('.tbin')
                    binary.unlink(missing_ok=True)
                    tensor.build(source, binary, target='sm_89', cache_dir=root/'cache')
                    with device.load(binary) as exe, ExitStack() as resources:
                        sig = exe.manifest['arguments']
                        sigs.append(sig)
                        if tag == 'old':
                            case = vocabulary_validation(kind,p,sig) if kind in ('argmax','embedding') else canonical_validation(kind,kwargs['parameters'],sig)
                            if kind == 'argmax':
                                v = case.arguments['x'].reshape(p['r'], p['c'])
                                v[0, [0, -1]] = 100
                                case.expected['out'][0] = 0
                            if kind == 'embedding':
                                case.arguments['tokens'][-1] = p['start'] - 1
                                case.expected['out'].reshape(p['r'],p['c'])[-1] = case.arguments['out'].reshape(p['r'],p['c'])[-1]
                        assert sig == sigs[0]
                        buffers = {}
                        for key in sig:
                            name = key['name']
                            buf = device.from_numpy(case.arguments[name])
                            buffers[name] = buf
                            resources.callback(buf.release)
                        exe.launch(*buffers.values())
                        device.synchronize()
                        outcomes.append({name: buf.to_numpy() for name, buf in buffers.items()})
                        kernels.append(hashlib.sha256(binary.read_bytes()).hexdigest())
                assert all(np.isfinite(a).all() for outputs in outcomes for a in outputs.values())
                bitwise = all(a.tobytes() == outcomes[1][name].tobytes() for name, a in outcomes[0].items())
                if kind in ('argmax','embedding'):
                    np.testing.assert_array_equal(outcomes[1]['out'], case.expected['out'])
                row = dict(module=module,factory=factory,parameters=p,bitwise=bitwise,old_tbin=kernels[0],new_tbin=kernels[1])
                metadata['checks'].append(row)
                print(json.dumps(row), flush=True)
    assert all(row['bitwise'] for row in metadata['checks'])
except BaseException:
    metadata['error'] = traceback.format_exc()
    rec.update(dict(schema='akas.benchmark.v1', metadata=metadata,results=[]),final=True,status='failed')
    print(rec.id, flush=True)
    raise
else:
    rec.update(dict(schema='akas.benchmark.v1',metadata=metadata,results=[]),final=True,status='passed')
    print(rec.id, flush=True)

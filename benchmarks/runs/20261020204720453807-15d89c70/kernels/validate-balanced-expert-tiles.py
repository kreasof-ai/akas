"""Real packed weights, independent sampled products, balanced/skewed routing."""
from contextlib import ExitStack
import json
from pathlib import Path
import traceback
import numpy as np
import tensor
from tensor.compiler.tuning import measure
from akas.artifacts.read import artifact_path, read_program
from akas.benchmarking.history import Recorder, write
from akas.checkpoints.gguf import GGUF
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.ops.moe.validation import tile_tables
from akas.quantization.weights.ggml.validation import decode_rows
from akas.memory.transfer import write as transfer

root = Path('build/lfm-moe/balanced-expert-tiles')
program = Path('build/lfm-moe/cuda-full/cuda/program')
checkpoint = Path('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf')
index = read_program(program)
manifest = json.loads((program / next(p['path'] for p in index['phases'] if p['phase'] == 'prefill' and p['chunk'] == 4096)).read_text())
meta = dict(runtime='akas', purpose='kernel', protocol='kernel-latency-v1', provider='cuda', target='sm_89',
    device=dict(name='NVIDIA L4', provider='cuda'), checkpoint=str(checkpoint),
    checkpoint_sha256='d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5', weight_format='q4_0',
    operand_dtype='float16', accumulator_dtype='float32',
    qualification='Original midpoint expert weight banks, synthetic activations with balanced and deliberately skewed valid routing. Independent F64 products check the first active route in every expert across all output columns; complete output tensors must match the qualified CUDA AOT bitwise. Includes routed scatter and inactive output sentinels; no model/serving performance claim.',
    reference_run_id='20261020182637613063-8a87f1a5')
rec = Recorder('benchmarks', meta, kind='algorithm', change='Measure smaller staging footprints and warp counts with real expert banks and skewed routing before selecting CUDA prefill geometry')
report = dict(schema='akas.benchmark.v1', metadata=meta, results=[])
passed = False
try:
    gguf = GGUF(checkpoint)
    with tensor.Device(provider='cuda') as device:
        for kind, depth, columns in (('gate', 2048, 1536), ('down', 1536, 2048)):
            p = dict(r=4096, top=4, experts=64, k=depth, o=columns, routed_input=kind == 'down')
            old = artifact_path(program, next(k for k in manifest['kernels'].values() if k['kind'] == 'quantized_experts' and k['parameters']['k'] == depth))
            names = sorted(n for n in gguf.tensors if n.endswith(f'ffn_{kind}_exps.weight'))
            name = names[len(names) // 2]
            assert gguf.tensors[name].type == 2
            with ExitStack() as owned:
                baseline = device.load(old)
                owned.callback(baseline.release)
                arrays = sample_arguments(baseline.manifest['arguments'])
                arrays['x'] *= 8
                arrays['w'][:] = gguf.packed(name)
                gguf.discard(name)
                arrays['out'].fill(1.25)
                buffers = {n: device.from_numpy(a) for n, a in arrays.items()}
                for buffer in buffers.values():
                    owned.callback(buffer.release)
                variants = [('original', baseline)]
                for path in sorted(root.glob(kind + '-*.tbin')):
                    module = device.load(path)
                    owned.callback(module.release)
                    variants.append((path.stem, module))
                meta.setdefault('weight_banks', {})[kind] = name
                for layout in ('balanced', 'skewed'):
                    if layout == 'balanced':
                        ids = (np.arange(4096)[:, None] + np.arange(4)[None, :]) % 64
                    else:
                        ids = np.column_stack((np.zeros(4096, dtype='int32'), np.ones(4096, dtype='int32'),
                            2 + np.arange(4096) % 62, 2 + (np.arange(4096) + 1) % 62))
                    arrays['counts'].fill(0)
                    arrays['routes'].fill(-1)
                    sampled = []
                    for expert in range(64):
                        owners, ranks = np.where(ids[:-1] == expert)
                        arrays['counts'][expert] = len(owners)
                        arrays['routes'][expert, :len(owners)] = owners * 4 + ranks
                        owner, rank = int(owners[0]), int(ranks[0])
                        source = arrays['x'][owner, rank] if p['routed_input'] else arrays['x'][owner]
                        source = source.astype('float16').astype('float64')
                        expected = np.empty(columns, dtype='float32')
                        for start in range(0, columns, 128):
                            stop = min(columns, start + 128)
                            weight = decode_rows(arrays['w'], 2, depth, expert * columns + start, expert * columns + stop).astype('float16').astype('float64')
                            expected[start:stop] = weight @ source
                        sampled.append((owner, rank, expected))
                    tables, rows = tile_tables(arrays['counts'], arrays['tile_experts'].shape[1])
                    arrays['tile_experts'][:], arrays['tile_rows'][:] = tables, rows
                    for field in ('counts', 'routes', 'tile_experts', 'tile_rows'):
                        transfer(device, buffers[field], arrays[field])
                    expected = None
                    for geometry, module in variants:
                        transfer(device, buffers['out'], arrays['out'])
                        args = tuple(buffers[a['name']] for a in module.manifest['arguments'])
                        plan = ExecutionPlan(device)
                        plan.bind(module, args)
                        graph = plan.capture(resources=tuple(buffers.values()) + (module,))
                        try:
                            graph.launch()
                            actual = buffers['out'].to_numpy()
                            for owner, rank, sample in sampled:
                                np.testing.assert_allclose(actual[owner, rank], sample, rtol=.002, atol=.0002)
                            if expected is None:
                                expected = actual.copy()
                            np.testing.assert_array_equal(actual.view('uint32'), expected.view('uint32'))
                            np.testing.assert_array_equal(actual[-1], arrays['out'][-1])
                            timing = measure(device, graph.launch, warmup=3, samples=5, repeats=2)
                        finally:
                            graph.close()
                        check = dict(kind=kind, layout=layout, geometry=geometry, bitwise=True,
                            independent_reference_rows=len(sampled), inactive_unchanged=True, timing=timing)
                        meta.setdefault('checks', []).append(check)
                        write(root / 'report.json', report)
                        print('PASS', kind, layout, geometry, timing['median_seconds'] * 1000, flush=True)
    passed = True
except BaseException:
    meta['traceback'] = traceback.format_exc()
    print(meta['traceback'], flush=True)
finally:
    for path in (__file__, 'build/lfm-moe/wide_expert_rows.py', 'build/lfm-moe/compile-balanced-expert-tiles.py'):
        rec.artifact(path, 'kernels/' + Path(path).name)
    for path in sorted(root.glob('*.tbin')):
        rec.artifact(path, 'kernels/' + path.name)
    write(root / 'report.json', report)
    rec.update(report, final=True, status='passed' if passed else 'failed')
    print('RETAINED', rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

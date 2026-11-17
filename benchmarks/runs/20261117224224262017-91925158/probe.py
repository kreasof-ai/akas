import hashlib
import json
import sys
import traceback
from pathlib import Path
from threading import Event, Thread

import numpy as np
import tensor
import torch
from akas.artifacts.write import checksum
from akas.benchmarking.history import Recorder
from akas.models.zonos3.model import SILENCE
from akas.models.zonos3.streaming import Zonos3StreamingEngine, Zonos3StreamingRuntime, _soft_limit_pcm, flush_codec_tail
from akas.models.zonos3.weights import NativeZonos3MiniCheckpoint
from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec
from zonos3.codec.istft_head_a_20261002.stream import ISTFTStream

root, ref = map(Path, sys.argv[1:3])
batched = '--batched' in sys.argv[3:]
bundle = root / 'akas-bundle'
codec_pt = Path('build/codec-artifacts/keep2_plain-9d1b6342.pt')
workload_path = Path('src/akas/benchmarking/workloads/speech-concurrent-v1.json')
text = json.loads(workload_path.read_text())['utterances'][0]['text']
metadata = {
    'runtime': 'akas', 'provider': 'cuda', 'weight_format': 'bf16',
    'decoder': 'keep2_plain', 'purpose': 'batched-slot-reuse-and-codec-diagnostic' if batched else 'c1-slot-reuse-and-codec-diagnostic',
    'protocol': 'zonos3-streaming-qualification-v1', 'program_sha256': checksum(bundle / 'streaming.json'),
    'oracle_source_run': '20261117163259599695-66d7e9b8', 'codec_sha256': checksum(codec_pt),
    'workload_sha256': checksum(workload_path), 'checks': [],
}
if batched:
    measurement = root / 'akas-diagnostics/runs/20261117223729127747-329927d0/run.json'
    assert json.loads(measurement.read_text())['metadata']['program_sha256'] == metadata['program_sha256']
    metadata['measurement_run'] = measurement.parent.name
rec = Recorder(root / 'akas-diagnostics', metadata, kind='qualification', change='FP32 codec agreement and BF16 sampled/greedy slot reuse', dependency_sources={'zonos3': ref})
rec.artifact(__file__, 'probe.py')
rec.artifact(bundle / 'streaming.json', 'program/streaming.json')
for path in bundle.glob('*/forward.json'):
    rec.artifact(path, f'program/{path.parent.name}/forward.json')
report = dict(schema='akas.benchmark.v1', metadata=metadata, results=[])

def check_pcm(label, actual, expected, initial=0):
    a, b = actual[initial:], expected
    assert a.shape == b.shape, (label, actual.shape, expected.shape, initial)
    assert np.isfinite(a).all()
    d = a.astype('float64') - b
    rms, peak = float(np.sqrt(np.mean(d*d))), float(np.max(np.abs(d)))
    row = dict(check=label, samples=len(a), rms=rms, max_abs=peak, rms_limit=5e-4, max_abs_limit=1e-2)
    metadata['checks'].append(row)
    print(json.dumps(row), flush=True)
    assert rms <= row['rms_limit'] and peak <= row['max_abs_limit'], row

def check_batches(rt, engine):
    for batch in (1, 4, 16, 64, 256):
        fingerprints = None
        for rep in (0, 1):
            done, errors = Event(), []
            audio = [hashlib.sha256() for _ in range(batch)]
            sizes, peaks = [0]*batch, [0.]*batch
            jobs, completed = [], []

            def chunk(data, idx):
                pcm = np.frombuffer(data, dtype='<f4')
                assert np.isfinite(pcm).all()
                audio[idx].update(data)
                sizes[idx] += len(pcm)
                peaks[idx] = max(peaks[idx], float(np.max(np.abs(pcm))))

            def complete(idx):
                completed.append(idx)
                if len(completed) == batch:
                    done.set()

            def on_error(message):
                errors.append(message)
                done.set()

            with engine._cond:
                for idx in range(batch):
                    temp = 0. if idx % 2 == 0 else .75
                    jobs.append(engine.submit(text, seed=1000 + idx, temperature=temp,
                        on_chunk=lambda data, i=idx: chunk(data, i),
                        on_done=lambda i=idx: complete(i), on_error=on_error))
            assert done.wait(75), f'C{batch} repetition {rep} timed out'
            assert not errors and len(set(completed)) == batch, errors
            assert not rt.pool.occupied.any(), f'C{batch} leaked slots'
            assert all(peak > 1e-5 for peak in peaks)
            assert all(size >= 44100 for size in sizes)
            samples = [
                (job.complete_frames(rt.config.codebooks) - 3) * 2940 for job in jobs
            ]
            assert sizes == samples, (sizes, samples)
            actual = [
                (job.step, size, pcm.hexdigest(), hashlib.sha256(np.asarray(job.delayed_rows).tobytes()).hexdigest())
                for job, size, pcm in zip(jobs, sizes, audio, strict=True)
            ]
            assert rep == 0 or actual == fingerprints, f'C{batch} changed on slot reuse'
            fingerprints = actual
            row = dict(case={'batch': batch, 'prompt': 50, 'generated': max(j.step for j in jobs)},
                       rep=rep, completed=len(completed), slot_reuse_bitwise=bool(rep),
                       min_audio_peak=min(peaks), fingerprints=actual, status='ok')
            report['results'].append(row)
            print(f'C{batch} rep {rep}: {len(completed)}/{batch} valid utterances', flush=True)

try:
    ckpt = NativeZonos3MiniCheckpoint(root / 'cache2/zonos3-mini/lm/zonos3_mini', codec_pt)
    prompt, _ = ckpt.prompt_encoded_frames(text)
    assert len(prompt) == 50 and len(text.split()) == 30 and len(text.encode()) == 177
    with tensor.Device() as dev, Zonos3StreamingRuntime(ckpt, bundle, dev, max_requests=256 if batched else 1) as rt:
        print('Runtime loaded', flush=True)
        torch.set_num_threads(1)
        blob = torch.load(codec_pt, map_location='cpu', weights_only=True)
        oracle = ISTFTCodec(blob['config']['model'], blob['config']['head'])
        oracle.load_state_dict(blob['state_dict'], strict=True)
        del blob
        stream = ISTFTStream(oracle.eval())
        slot = 7
        sil = np.tile(np.asarray(SILENCE, dtype='int32'), (3, 1))
        rt.reset_slots([slot])
        primed, = rt.run_codec_batch([(slot, 0, sil)])
        check_pcm('prime-slot-7', primed, _soft_limit_pcm(stream.decode(sil)), initial=1470)
        rng = np.random.default_rng(1234)
        pos = 3
        for idx, size in enumerate((1, 2, 4, 8, 15, 7, 13, 15)):
            codes = rng.integers(0, 4096, size=(size, 10), dtype='int32')
            if idx == 0:
                codes[:, 1:] = -1
            actual, = rt.run_codec_batch([(slot, pos, codes)])
            check_pcm(f'codec-slot-7-pos-{pos}-frames-{size}', actual, _soft_limit_pcm(stream.decode(codes)))
            pos += size
        ola = rt._states['istft_ola_state'].to_numpy()[slot]
        rs = rt._states['istft_rs_state'].to_numpy()[slot]
        check_pcm('flush-slot-7', flush_codec_tail(ola, rs, rt._mixer, rt._resample), _soft_limit_pcm(stream.flush()))
        rt.reset_slots([slot])
        reused, = rt.run_codec_batch([(slot, 0, sil)])
        assert np.array_equal(primed, reused), 'slot reuse changed the silence prime'
        print('Mapped slot 7 reset/reuse is bitwise identical', flush=True)
        if batched:
            for group in ((255, 32, 111, 9), (255, *range(9, 24))):
                rt.prime_slots(group)
                streams = [ISTFTStream(oracle) for _ in group[:2]]
                for other in streams:
                    other.decode(sil)
                positions = [3] * len(group)
                for frames in (1, 4, 15, 3, 15, 15):
                    rows = [rng.integers(0, 4096, size=(max(1, frames - i % 4), 10), dtype='int32')
                            for i in range(len(group))]
                    results = rt.run_codec_batch(list(zip(group, positions, rows, strict=True)))
                    for j, other in enumerate(streams):
                        check_pcm(f'codec-batch-{len(group)}-slot-{group[j]}-pos-{positions[j]}',
                                  results[j], _soft_limit_pcm(other.decode(rows[j])))
                    positions = [pos + len(codes) for pos, codes in zip(positions, rows, strict=True)]
        engine = Zonos3StreamingEngine(rt)
        worker = Thread(target=engine.run_forever, daemon=True)
        worker.start()
        try:
            if batched:
                check_batches(rt, engine)
            expected = {}
            for temperature in (() if batched else (0., .75, 0., .75)):
                done, chunks, errors = Event(), [], []
                def on_error(message):
                    errors.append(message)
                    done.set()
                job = engine.submit(text, temperature=temperature, on_chunk=chunks.append, on_done=done.set, on_error=on_error)
                assert done.wait(90), 'utterance timed out'
                assert not errors, errors
                data = b''.join(chunks)
                pcm = np.frombuffer(data, dtype='<f4')
                assert np.isfinite(pcm).all() and len(pcm) >= 44100 and np.max(np.abs(pcm)) > 1e-5
                fingerprint = (hashlib.sha256(data).hexdigest(), hashlib.sha256(np.asarray(job.delayed_rows).tobytes()).hexdigest())
                assert temperature not in expected or fingerprint == expected[temperature], 'nondeterministic reuse'
                expected[temperature] = fingerprint
                row = dict(case={'batch': 1, 'prompt': 50, 'generated': job.step}, temperature=temperature, samples=len(pcm), duration_s=len(pcm)/44100, chunks=len(chunks), peak=float(np.max(np.abs(pcm))), rms=float(np.sqrt(np.mean(pcm**2))), audio_sha256=fingerprint[0], delayed_rows_sha256=fingerprint[1], generated_rows=job.step, status='ok')
                report['results'].append(row)
                print(json.dumps(row), flush=True)
        finally:
            engine.stop()
            worker.join()
except BaseException:
    metadata['error'] = traceback.format_exc()
    rec.update(report, final=True, status='failed')
    print(f'Recorded {rec.id}', flush=True)
    raise
else:
    rec.update(report, final=True, status='passed')
    print(f'Recorded {rec.id}', flush=True)

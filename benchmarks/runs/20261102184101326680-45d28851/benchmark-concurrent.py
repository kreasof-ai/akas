"""Deploy the existing L40S streamed benchmark contract on one H200 at a time."""
from pathlib import Path
import modal
from tools.h200_modal import app, volume, vimage, simage, aimage, ROOT, MODEL, REV


def baseline_worker(runtime):
    import os, subprocess, sys, time
    os.chdir('/workspace/akas')
    os.environ.update(OMP_NUM_THREADS='2', FLASHINFER_WORKSPACE_BASE='/data/cache/flashinfer',
                      VLLM_CACHE_ROOT='/data/cache/vllm', TRITON_CACHE_DIR='/data/cache/triton',
                      TORCHINDUCTOR_CACHE_DIR='/data/cache/inductor')
    checkpoint = Path('build/checkpoints/lfm2.5-24b-a2b-fp8')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists(): checkpoint.symlink_to('/data/checkpoint')
    python = Path(f'build/baselines/{runtime}/bin/python')
    python.parent.mkdir(parents=True, exist_ok=True)
    if not python.exists(): python.symlink_to(sys.executable)
    command = [sys.executable, '-u', '-m', 'tools.baselines', runtime,
               'lfm2.5-24b-a2b-fp8', '--format', 'fp8', '--optimized', '--streaming',
               '--workload', 'concurrent-prefill-v1', '--seconds', '900',
               '--history', '/data/standard-benchmarks']
    print('STANDARD COMMAND', command, flush=True)
    tick = time.monotonic()
    result = subprocess.run(command)
    volume.commit()
    return dict(runtime=runtime, returncode=result.returncode, elapsed_seconds=time.monotonic()-tick)


@app.function(image=vimage.add_local_dir(ROOT/'benchmarks/models', '/workspace/akas/benchmarks/models'),
              gpu='H200', volumes={'/data': volume}, timeout=1100, cpu=4, memory=65536,
              scaledown_window=2, max_containers=1)
def standard_vllm():
    return baseline_worker('vllm')


@app.function(image=simage.add_local_dir(ROOT/'benchmarks/models', '/workspace/akas/benchmarks/models'),
              gpu='H200', volumes={'/data': volume}, timeout=1100, cpu=4, memory=65536,
              scaledown_window=2, max_containers=1)
def standard_sglang():
    return baseline_worker('sglang')


@app.function(image=aimage, gpu='H200', volumes={'/data': volume}, timeout=400,
              cpu=4, memory=65536, scaledown_window=2, max_containers=1)
def standard_akas():
    import asyncio, hashlib, json, os, time
    from dataclasses import asdict
    import tensor
    from akas import Engine, EngineConfig
    from akas.models.lfm2 import LFM2MoE
    from akas.benchmarking.cases import Case
    from akas.benchmarking.history import Recorder, write
    from akas.benchmarking.runner import summarize
    from akas.benchmarking.streaming import engine_burst
    from akas.benchmarking.workload import synthetic_prompts, prompt_digest, prompt_seed
    from akas.cli import checkpoint_digest
    os.chdir('/workspace/akas')
    os.environ['OMP_NUM_THREADS'] = '2'
    program = Path('/data/programs/final-best')
    config = EngineConfig(prefill_chunk=2048, prefill_requests=256,
                          prefill_token_budget=256*2048, decode_prefill_token_budget=256)
    manifest = json.loads((program/'program.json').read_text())
    metadata = dict(runtime='akas', purpose='concurrent-fp8-diagnostic', model=MODEL,
                    model_revision=REV, weight_format='fp8', provider='cuda', target='sm_90a',
                    mode='ar', kv_dtype='float16', kv_bits=16, prefix_cache=False,
                    speculative_decoding=False, cpu_offload=False, early_eos=False,
                    protocol='client-stream-burst-v1', warmup=1, repeats=3,
                    full_model_numerical_qualification='open', model_throughput_qualified=False,
                    engine_config=asdict(config), execution_policy=manifest.get('policy', {}),
                    checkpoint_sha256=checkpoint_digest('/data/checkpoint'),
                    program_sha256=hashlib.sha256((program/'program.json').read_bytes()).hexdigest())
    rec = Recorder('/data/standard-benchmarks', metadata, kind='benchmark',
                   change='Remeasure the current H200 candidate with the retained L40S streamed protocol',
                   dependency_sources={'tensor': Path('/workspace/tensor')})
    rec.artifact(Path(__file__), 'benchmark-concurrent.py')
    print('STARTED', rec.id, flush=True)
    tick = time.monotonic()
    results = []
    status = 'failed'
    try:
        with tensor.Device() as device:
            metadata['device'] = device.info
            with Engine.from_checkpoint(LFM2MoE, '/data/checkpoint', program, device,
                                        config=config, checkpoint_options={'tokenizer': False}) as engine:
                for batch in [1, 4, 16, 64, 256]:
                    prompts = synthetic_prompts(engine.model.config.vocab, batch, 2048)
                    print('WARMUP', batch, flush=True)
                    asyncio.run(engine_burst(engine, prompts, 128))
                    samples, outputs = [], []
                    for repeat in range(3):
                        ids, metrics = asyncio.run(engine_burst(engine, prompts, 128))
                        metrics['decode_tokens_per_second_per_request'] = metrics['decode_tokens_per_second']/batch
                        metrics['inter_token_seconds'] = metrics['tpot_seconds']
                        metrics['wall_seconds'] = max(s['arrivals_seconds'][-1] for s in metrics['request_streams'])
                        samples.append(metrics)
                        outputs.append(ids)
                        print('SAMPLE', batch, repeat, metrics['prefill_tokens_per_second'], metrics['decode_tokens_per_second'], flush=True)
                    deterministic = all(value == outputs[0] for value in outputs)
                    counts = all(len(ids) == batch and all(len(seq) == 128 for seq in ids) for ids in outputs)
                    row = dict(case=Case(batch, 2048, 128).metadata(), status='ok' if counts and deterministic else 'error',
                               seed=prompt_seed(batch, 2048), prompt_source='synthetic', prompt_sha256=prompt_digest(prompts),
                               prompt_tokens=batch*2048, generated_tokens=batch*128, decode_tokens=batch*127,
                               full_model_numerical_qualification='open', model_throughput_qualified=False,
                               samples=samples, summary={key:summarize([s[key] for s in samples]) for key in
                                   ('ttft_seconds', 'prefill_tokens_per_second', 'decode_tokens_per_second',
                                    'decode_tokens_per_second_per_request', 'inter_token_seconds', 'wall_seconds')},
                               output_ids=outputs[0], deterministic=deterministic, output_count=counts,
                               scheduler=dict(engine.scheduler.stats))
                    results.append(row)
                    rec.update(dict(metadata=metadata, results=results))
        rec.program(program)
        status = 'passed' if all(row['status']=='ok' for row in results) else 'failed'
    finally:
        report = dict(schema='akas.benchmark.v1', metadata=metadata, results=results)
        raw = rec.directory/'raw.json'
        write(raw, report)
        rec.artifact(raw, 'raw.json')
        record = rec.update(report, final=True, status=status)
        volume.commit()
        print('RUN', rec.id, record['status'], flush=True)
    return dict(runtime='akas', run_id=rec.id, returncode=int(record['status']!='passed'),
                elapsed_seconds=time.monotonic()-tick)


@app.local_entrypoint(name='remeasure')
def main():
    import json, time
    ledger_path = ROOT/'build/h200/cost-ledger.json'
    rate = .001261 + 4*.0000131 + 64*.00000222
    for runtime, function, bound in [('vllm', standard_vllm, 1100),
                                      ('sglang', standard_sglang, 1100),
                                      ('akas', standard_akas, 400)]:
        ledger = json.loads(ledger_path.read_text())
        if sum(row['conservative_usd'] for row in ledger) + bound*rate > 28:
            raise RuntimeError('Campaign budget exceeded')
        row = dict(action='standard-'+runtime, conservative_usd=bound*rate,
                   status='reserved', started_at=time.time())
        ledger.append(row)
        ledger_path.write_text(json.dumps(ledger, indent=2))
        tick = time.monotonic()
        try:
            result = function.remote()
            row.update(status='completed', result=result)
            print('STANDARD RESULT', json.dumps(result), flush=True)
        finally:
            row['elapsed_seconds'] = time.monotonic()-tick
            if row['status']=='completed': row['conservative_usd']=(row['elapsed_seconds']+30)*rate
            ledger_path.write_text(json.dumps(ledger, indent=2))

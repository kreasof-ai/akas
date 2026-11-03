"""One frozen baseline run per upstream server, launched with only a model path."""
from pathlib import Path
import modal
from tools.h200_modal import ROOT, MODEL, REV, volume, vimage, simage

app = modal.App('akas-h200-default-baselines')


def measure_default_server(runtime):
    import asyncio
    import csv
    import importlib.metadata
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    import traceback
    import urllib.error
    import urllib.request
    import uuid
    from datetime import datetime, timezone
    import aiohttp
    from akas.benchmarking.history import Recorder, write
    from akas.benchmarking.streaming import stream_burst
    from akas.cli import checkpoint_digest
    import tools.baselines.common as common

    os.chdir('/workspace/akas')
    # Remove the earlier benchmark's threading and tuning overrides.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MAX_JOBS',
                'SGLANG_ENABLE_DELAY_SAMPLE', 'AKAS_GPU_EVENT_DIR', 'AKAS_GPU_EVENT_CONTROL'):
        os.environ.pop(key, None)
    # Persist compiled code; these paths do not change engine settings.
    os.environ.update(FLASHINFER_WORKSPACE_BASE='/data/cache/flashinfer',
                      VLLM_CACHE_ROOT='/data/cache/vllm', TRITON_CACHE_DIR='/data/cache/triton',
                      TORCHINDUCTOR_CACHE_DIR='/data/cache/inductor')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = Path('/data/default-baselines')/f'{stamp}-{runtime}'
    directory.mkdir(parents=True)
    checkpoint = '/data/checkpoint'
    command = (['vllm', 'serve', checkpoint] if runtime == 'vllm' else
               [sys.executable, '-m', 'sglang.launch_server', '--model-path', checkpoint])
    origin = 'http://127.0.0.1:' + ('8000' if runtime == 'vllm' else '30000')
    config = dict(shapes=[[b, 2048] for b in [1, 4, 16, 64, 256]], output_tokens=128,
                  warmup=1, repeats=3, vocab=json.loads(Path(checkpoint+'/config.json').read_text())['vocab_size'])
    write(directory/'workload.json', config)
    gpu = next(csv.reader(subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,compute_cap,uuid,driver_version,memory.total',
        '--format=csv,noheader,nounits'], text=True).splitlines()))
    metadata = dict(runtime=runtime, purpose='baseline', model=MODEL, model_revision=REV,
                    provider='cuda', target='sm_'+gpu[1].strip().replace('.', ''), mode='ar',
                    device=dict(name=gpu[0].strip(), uuid=gpu[2].strip(), driver=gpu[3].strip(), memory_mib=int(gpu[4])),
                    command=command, configuration='upstream server defaults; model path only',
                    engine_overrides={}, runtime_version=importlib.metadata.version(runtime),
                    protocol='baseline-stream-burst-v1', weight_format='fp8', warmup=1, repeats=3,
                    checkpoint_sha256=checkpoint_digest(checkpoint),
                    sampling=dict(temperature=0, ignore_eos=True, output_tokens=128),
                    cache_workload='cold prompts; engine prefix caching remains at its default',
                    transport='default HTTP server; common client-stream-burst-v1 collector')
    rec = Recorder('/data/default-benchmarks', metadata, kind='baseline',
                   change='Measure upstream default server configuration once; no tuning')
    rec.artifact(Path(__file__), 'benchmark.py')
    rec.artifact(directory/'workload.json', 'workload.json')
    common._output = directory/'raw.json'
    report = dict(runtime=runtime, version=metadata['runtime_version'], command=command, cases=[])
    process = None
    log = (directory/'server.log').open('w')
    tick = time.monotonic()
    deadline = tick + 1100
    status = 'failed'

    def request(path, *, post=False):
        req = urllib.request.Request(origin+path, method='POST' if post else 'GET')
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read()

    try:
        print('DEFAULT SERVER', runtime, command, 'RUN', rec.id, flush=True)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            if process.poll() is not None:
                raise RuntimeError(f'Default server exited with {process.returncode}; see server.log')
            if time.monotonic() > deadline:
                raise TimeoutError('Default server startup exceeded the fixed job deadline')
            try:
                request('/health')
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        print('READY', runtime, round(time.monotonic()-tick, 2), flush=True)
        info = json.loads(request('/server_info')) if runtime == 'sglang' else json.loads(request('/v1/models'))
        write(directory/'server-info.json', info)
        metadata['server_info'] = info
        incremental = info.get('incremental_streaming_output', False)
        serial = 0

        async def burst(prompts, salt):
            timeout = aiohttp.ClientTimeout(total=max(1, min(240, deadline-time.monotonic())))
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout) as client:
                async def generate(index, prompt):
                    if runtime == 'vllm':
                        body = dict(model=checkpoint, prompt=prompt, max_tokens=128,
                                    temperature=0, ignore_eos=True, stream=True,
                                    return_token_ids=True, cache_salt=salt)
                        endpoint = '/v1/completions'
                    else:
                        body = dict(input_ids=prompt, sampling_params=dict(temperature=0,
                                    max_new_tokens=128, ignore_eos=True), stream=True)
                        endpoint = '/generate'
                    tokens = []
                    async with client.post(origin+endpoint, json=body) as response:
                        if response.status != 200:
                            raise ValueError(f'{runtime} HTTP {response.status}: {(await response.text())[:1500]}')
                        async for line in response.content:
                            line = line.strip()
                            if not line.startswith(b'data:'):
                                continue
                            payload = line[5:].strip()
                            if payload == b'[DONE]':
                                continue
                            row = json.loads(payload)
                            if runtime == 'vllm':
                                choices = row.get('choices', [])
                                if not choices:
                                    continue
                                choice = choices[0]
                                if choice.get('prompt_token_ids') is not None and choice['prompt_token_ids'] != prompt:
                                    raise ValueError('Server changed the supplied prompt IDs')
                                tokens.extend(choice.get('token_ids') or [])
                                details = dict(finish_reason=choice.get('finish_reason'))
                            else:
                                current = row.get('output_ids', [])
                                tokens = tokens + current if incremental else list(current)
                                details = row.get('meta_info', {})
                            yield list(tokens), details
                return await stream_burst(prompts, generate, 128)

        def invoke(batch, prefix):
            nonlocal serial
            if time.monotonic() > deadline:
                raise TimeoutError('Fixed measurement deadline exceeded')
            serial += 1
            if runtime == 'sglang':
                request('/flush_cache', post=True)
            # vLLM's reset endpoint is development-only. Per-burst cache salts
            # keep the normal production server unchanged and prevent reuse.
            salt = f'akas-default-cold-{rec.id}-{serial}-{uuid.uuid4().hex}'
            return asyncio.run(burst(common.synthetic_prompts(config['vocab'], batch, prefix), salt))

        common.measurements(config, invoke, report)
        status = 'passed' if len(report['cases']) == 5 and all(r['status']=='passed' for r in report['cases']) else 'failed'
        try:
            (directory/'metrics.txt').write_bytes(request('/metrics'))
        except (urllib.error.URLError, TimeoutError):
            pass
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        log.close()
        report['status'] = status
        common.save(report)
        metadata['error'] = report.get('error')
        rows = [dict(case=dict(batch=r['batch'], prompt=r['prefix'], generated=r['output'], depth=0, mixed=False),
                     status='ok' if r['status']=='passed' else 'error', samples=r.get('samples', []),
                     correctness=dict(output_count=r.get('output_count', False), deterministic=r.get('deterministic')),
                     output_ids=r.get('output_ids'), error=r.get('error')) for r in report['cases']]
        for name in ('raw.json', 'server.log', 'server-info.json', 'metrics.txt'):
            if (directory/name).exists():
                rec.artifact(directory/name, name)
        record = rec.update(dict(schema='akas.benchmark.v1', metadata=metadata, results=rows), final=True, status=status)
        volume.commit()
        print('FINISHED', runtime, rec.id, record['status'], flush=True)
    return dict(runtime=runtime, run_id=rec.id, status=record['status'], directory=str(directory),
                elapsed_seconds=time.monotonic()-tick)


@app.function(image=vimage, gpu='H200', volumes={'/data': volume}, timeout=1200,
              cpu=8, memory=65536, scaledown_window=2, max_containers=1)
def default_vllm():
    return measure_default_server('vllm')


@app.function(image=simage, gpu='H200', volumes={'/data': volume}, timeout=1200,
              cpu=8, memory=65536, scaledown_window=2, max_containers=1)
def default_sglang():
    return measure_default_server('sglang')


@app.local_entrypoint()
def main():
    import json
    import time
    ledger_path = ROOT/'build/h200/cost-ledger.json'
    rate = .001261 + 8*.0000131 + 64*.00000222
    for runtime, function in [('vllm', default_vllm), ('sglang', default_sglang)]:
        ledger = json.loads(ledger_path.read_text())
        if sum(row['conservative_usd'] for row in ledger) + 1200*rate > 28:
            raise RuntimeError('Campaign budget exceeded')
        row = dict(action='default-server-'+runtime, conservative_usd=1200*rate,
                   status='reserved', started_at=time.time())
        ledger.append(row)
        ledger_path.write_text(json.dumps(ledger, indent=2))
        tick = time.monotonic()
        try:
            result = function.remote()
            row.update(status='completed', result=result)
            print('FROZEN DEFAULT RESULT', json.dumps(result), flush=True)
        finally:
            row['elapsed_seconds'] = time.monotonic()-tick
            if row['status']=='completed':
                row['conservative_usd'] = (row['elapsed_seconds']+30)*rate
            ledger_path.write_text(json.dumps(ledger, indent=2))

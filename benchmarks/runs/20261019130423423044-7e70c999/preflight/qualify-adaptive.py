import json
from pathlib import Path
import sys
import os
import subprocess
import time
import tensor
from akas.compilation.compile import compile_program
from akas.compilation.tuning import SearchSettings
from akas.models.lfm2.model import LFM2MoE

provider = sys.argv[1]
started = time.monotonic()
try:
    checkpoint = LFM2MoE.checkpoint_type('build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf', tokenizer=False)
    target = 'sm_89' if provider == 'cuda' else 'webgpu-portable-v1'
    from tensor.compiler.search import ScheduleProfile
    seed_path = Path(f'build/lfm-moe/{provider}-seed-profile.json')
    schedules = ScheduleProfile(json.loads(seed_path.read_text())) if seed_path.is_file() else None
    with tensor.Device(provider=provider, max_buffer_size=268435456 if provider == 'webgpu' else None) as device:
        compile_program(checkpoint, f'build/lfm-moe/{provider}-512', definition=LFM2MoE.compose,
                        geometries=[(1, 1), (1, 32), (1, 256), (1, 512)], max_requests=1, max_context=131072,
                        provider=provider, target=target, device=device,
                        model_options=dict(LFM2MoE.model_defaults), schedules=schedules,
                        search_settings=SearchSettings(budget=1, width=1, warmup=0, samples=1, repeats=1))
    program_root = Path(f'build/lfm-moe/{provider}-512')
    manifest = json.loads((program_root/'program.json').read_text())
    for phase in manifest['phases']:
        forward = json.loads((program_root/phase['path']).read_text())
        for op in forward.get('tuning', {}).get('operations', []):
            expected = {**op['selected']['schedule'], 'implementation': op['selected']['implementation']}
            assert forward['kernels'][op['key']]['schedule'] == expected, f"Compiled schedule differs from qualified winner: {op['kind']}"
            if schedules is not None:
                seeded = schedules.select(op['kind'], op['parameters'], provider=provider, target=target)
                baseline = next(row['schedule'] for row in forward['tuning']['context']['baseline'] if row['key']==op['key'])
                assert all(baseline.get(k)==v for k,v in seeded.items()), f"Seed schedule was ignored: {op['kind']}"
    print('Compiled schedules match qualified winners and supplied seeds', flush=True)
finally:
    peak = next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:'))
    print('RESOURCE', json.dumps(dict(provider=provider, peak_rss_bytes=int(peak)*1024, seconds=time.monotonic()-started)), flush=True)

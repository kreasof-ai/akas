import json
from pathlib import Path

data = json.loads(Path('build/vulkan-tune/selected-archive.json').read_text())
for entry in data['entries']:
    kind, p = entry['operation'], entry['parameters']
    if kind in ('attention_adaptive_partial', 'attention_merge'):
        p['splits'] = 128
        entry['schedule'] = (dict(implementation='attention.webgpu.packed_decode', granule=128, key_rows=32)
                             if kind == 'attention_adaptive_partial' else
                             dict(implementation='attention.webgpu.stream_merge'))
    if kind not in ('linear', 'linear_add', 'ffn') or p.get('r', 1) == 1 or p.get('type') not in (1, 2):
        continue
    if kind == 'ffn' or p['o'] == 3 * p['k']:
        p['half_accum'] = True
        p['type'] = 1 if p['r'] >= 128 else 2
        layout = (1 if kind == 'ffn' else 2) if p['r'] >= 128 else (0 if kind == 'ffn' else 3)
        entry['schedule'] = dict(implementation='linear.webgpu.short_half', half_native=True,
                                 half_layout=layout, half_pad=1)
    else:
        p.update(type=2, q16=True, integer=dict(fixed_residual=False))
        layout = (2 if p['k'] == 10752 else 1 if p['o'] >= 2048 else 0) if p['r'] >= 128 else 0
        entry['schedule'] = dict(implementation='linear.webgpu.two_component', integer_layout=layout)
Path('build/vulkan-next/mixed-profile.json').write_text(json.dumps(data, indent=2))
Path('build/vulkan-next/policy-mixed.json').write_text(json.dumps(dict(
    attention=dict(splits=128, adaptive_splits=True), linear=dict(stage_weights=True)), indent=2))

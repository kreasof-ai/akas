from pathlib import Path
import hashlib
import json
import subprocess

repo = Path('D:/Workspace/kreasof-ai/github/tensor')
out = Path('build/vulkan-tune/tensor-archive')
out.mkdir(parents=True, exist_ok=True)
revision = subprocess.check_output(['git', 'rev-parse', 'd5ee508'], cwd=repo, text=True).strip()
files = [
    'docs/research/lfm2-2.6b-q4_0-parity.md',
    'docs/research/lfm2-2.6b-prefill-1k.md',
    'docs/research/data/lfm2-2.6b-q4_0-parity.json',
    'docs/research/data/lfm2-2.6b-prefill-1k.json',
    'packages/tensor-llm/src/tensor_llm/model.py',
]
retained = []
for name in files:
    data = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=repo)
    destination = out / Path(name).name
    destination.write_bytes(data)
    retained.append(dict(source=name, file=destination.name, sha256=hashlib.sha256(data).hexdigest()))
(out / 'provenance.json').write_text(json.dumps(dict(
    repository=str(repo), revision=revision, files=retained,
    source_checkpoint_sha256='a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03',
    target_checkpoint_sha256='e1a61bf937bc60726e18626e97f7ee9bfd2574d95744c2ed909de98b78006fbe',
    timing_note='Historical QAD Q4_0, context 512, forced-token host-logit calls; current standard Q4_0, context 131072, greedy native-token-step-v2. Historical rates are not directly comparable.',
    transfer='F32 packed decode schedules, subgroup RMS, and shared padding/grid placement only. Approximate short-F16 and Q8 prefill arithmetic is not enabled.'), indent=2))

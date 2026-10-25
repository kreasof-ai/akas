from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tilelang
from akas.compilation.selection import source
from akas.compilation.emit import build_artifact

root = Path('build/vulkan-next/debug') / (sys.argv[1] if len(sys.argv) > 1 else 'production')
root.mkdir(parents=True, exist_ok=True)
p = dict(r=1, h=32, kh=8, d=64, capacity=131072, splits=128,
         optimized=True, provider='webgpu', kv_dtype='float16', unit_scales=True,
         granule=128, key_lanes=8, value_unroll=8, value_chains=4)
if len(sys.argv) > 1:
    p.update(r=2, h=4, kh=1, capacity=int(sys.argv[1]))
entry = root/'kernel.py'
entry.write_text(source('attention_adaptive_partial', p, contract='reference',
    provider='webgpu', target='webgpu-portable-v1',
    preferred='attention.webgpu.two_pass_decode').render())
original = tilelang.transform.StorageRewrite
def debug():
    operation = original()
    def run(mod):
        (root/'before-storage.py').write_text(mod.script())
        return operation(mod)
    return run
tilelang.transform.StorageRewrite = debug
build_artifact(entry, root/'kernel.tbin', provider='webgpu', target='webgpu-portable-v1')

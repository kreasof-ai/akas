import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from akas.compilation.selection import source
from akas.compilation.emit import build_artifact
from tensor.artifacts.format import read_artifact

root = Path('build/vulkan-next/native-half-v3')
root.mkdir(parents=True, exist_ok=True)
entry, artifact = root/'kernel.py', root/'kernel.tbin'
entry.write_text(source('ffn', dict(r=32,k=256,o=128,ggml=True,type=1,half_accum=True),
    contract='reference',provider='webgpu',target='webgpu-portable-v1',
    preferred='linear.webgpu.short_half').render())
build_artifact(entry, artifact, provider='webgpu',target='webgpu-portable-v1')
_, files = read_artifact(artifact)
(root/'kernel.wgsl').write_bytes(files['kernel.wgsl'])
from akas.kernels.registry import validation
from tools.context_sweep import vulkan_device
meta, _ = read_artifact(artifact)
case = validation('ffn', dict(r=32,k=256,o=128,ggml=True,type=1,half_accum=True),meta['arguments'])
with vulkan_device() as device:
    arrays = [device.from_numpy(case.arguments[arg['name']]) for arg in meta['arguments']]
    with device.load(artifact) as executable:
        executable.launch(*arrays)
        case.comparators['out'](arrays[-1].to_numpy(),case.expected['out'])
    for array in arrays:
        array.release()
print('typed half pairs GPU reference passed',flush=True)

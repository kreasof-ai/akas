"""Prepare a bounded loop-unroll/column-reuse experiment and cold-cache flush."""
from pathlib import Path
from akas.compilation.emit import build_artifact

root = Path('build/lfm-moe/webgpu-expert-decode')
root.mkdir(exist_ok=True)
geometries = [(1, 1), (1, 4), (1, 8), (2, 4), (2, 8), (4, 4)]
for kind, depth, columns, experts in (('gate', 2048, 1536, 64), ('down', 1536, 2048, 64),
                                    ('small-gate', 256, 37, 6), ('small-down', 256, 37, 6)):
    p = dict(r=1, top=4, experts=experts, k=depth, o=columns, input_dtype='float32',
             round_inputs=False, routed_input=kind.endswith('down'),
             weight_parts=({'rows': experts * columns, 'format': 'ggml', 'options': {'encoding': 2}},))
    options = geometries if experts == 64 else [(2, 8)]
    if experts == 6:
        options = [('original', 1), *options]
    for width, unroll in options:
        path = root / f'{kind}-{width}-{unroll}.py'
        declaration = ('from akas.ops.moe.webgpu import warp_experts as factory' if width == 'original'
                       else 'from wgpu_expert_decode import factory')
        axes = {} if width == 'original' else dict(columns_per_warp=width, unroll=unroll)
        path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\n'+declaration+'\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),'+repr(axes)+')}\n')
        build_artifact(path, path.with_suffix('.tbin'), provider='webgpu', target='webgpu-portable-v1')
        print('BUILT', path.name, flush=True)
path = root / 'flush.py'
path.write_text('from akas.kernels.kernel import Tensor,kernel,output\nimport tilelang.language as T\n@kernel\ndef compute()->Tensor((33554432,),"uint32"):\n out=output((33554432,),"uint32")\n with T.Kernel(32768,threads=128) as bx:\n  for i in T.Parallel(1024):\n   out[bx*1024+i]=T.cast(bx*1024+i,"uint32")\n return out\ndef tensor_export():\n return {"kernel":compute}\n')
build_artifact(path, path.with_suffix('.tbin'), provider='webgpu', target='webgpu-portable-v1')
print('BUILT flush', flush=True)

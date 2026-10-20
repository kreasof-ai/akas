from pathlib import Path
from akas.compilation.emit import build_artifact

root = Path('build/lfm-moe/router-projection')
root.mkdir(exist_ok=True)
for kind, p in (('full', dict(r=1, k=2048, o=64)),
                ('small', dict(r=3, k=257, o=7, output_stride=15, output_offset=3, matrix_output=True))):
    choices = [(4, 32), (1, 32), (1, 64), (1, 128), (1, 256), (2, 128), (2, 256)] if kind == 'full' else [(1, 128)]
    for columns, threads in choices:
        axes = dict(columns_per_group=columns, threads=threads)
        path = root / f'{kind}-{columns}-{threads}.py'
        path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom router_projection import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("linear",'+repr(p)+'),'+repr(axes)+')}\n')
        build_artifact(path, path.with_suffix('.tbin'), provider='webgpu', target='webgpu-portable-v1')
        print('BUILT', path.name, flush=True)

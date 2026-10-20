from pathlib import Path
import json
from akas.compilation.emit import build_artifact

root = Path('build/lfm-moe/parallel-argmax')
root.mkdir(exist_ok=True)
for rows, columns, block in ((1, 128000, 1024), (2, 2051, 512)):
    p = dict(r=rows, c=columns, block=block)
    for name in ('partial', 'merge'):
        path = root / f'{rows}-{columns}-{block}-{name}.py'
        path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom parallel_argmax import '+name+'\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":'+name+'(KernelSpec("argmax",'+repr(p)+'),{})}\n')
        build_artifact(path, path.with_suffix('.tbin'), provider='webgpu', target='webgpu-portable-v1')
        print('BUILT', path.name, flush=True)

"""Compare smaller staging footprints and warp counts with the qualified CUDA geometry."""
from pathlib import Path
from akas.compilation.emit import build_artifact

root = Path('build/lfm-moe/balanced-expert-tiles')
root.mkdir(exist_ok=True)
choices = [(64, 128, 128, 128, 1), (64, 128, 128, 128, 2),
           (64, 128, 128, 256, 1), (64, 128, 64, 128, 1),
           (64, 128, 64, 128, 2), (128, 128, 64, 256, 1),
           (128, 128, 64, 256, 2), (32, 128, 128, 128, 1)]
for kind, depth, columns in (('gate', 2048, 1536), ('down', 1536, 2048)):
    for bm, bn, bk, threads, stages in choices:
        p = dict(r=4096, top=4, experts=64, k=depth, o=columns, input_dtype='float32',
                 round_inputs=True, tile_map=True, routed_input=kind == 'down',
                 weight_parts=({'rows': 64 * columns, 'format': 'ggml', 'options': {'encoding': 2}},))
        axes = dict(block_m=bm, columns=bn, block_k=bk, threads=threads, stages=stages)
        path = root / f'{kind}-{bm}-{bn}-{bk}-{threads}-{stages}.py'
        path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom wide_expert_rows import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("quantized_experts",'+repr(p)+'),'+repr(axes)+')}\n')
        build_artifact(path, path.with_suffix('.tbin'), provider='cuda', target='sm_89')
        print('BUILT', path.name, flush=True)

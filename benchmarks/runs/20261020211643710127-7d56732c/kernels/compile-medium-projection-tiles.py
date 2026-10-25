"""Compile larger prefill tiles through the unchanged public CUDA factory."""
from pathlib import Path
from akas.compilation.emit import build_artifact
from akas.compilation.selection import selected_source
from akas.kernels.definition import definition
from akas.benchmarking.history import write

root = Path('build/lfm-moe/medium-projection-tiles')
root.mkdir(exist_ok=True)
choices = [(64,64,64,128,2),(128,128,64,256,1),
           (64,256,32,256,2),(128,128,32,256,2)]
records = []
for name, rows, depth, columns in (
    (name+'-r'+str(rows),rows,depth,columns)
    for rows in (512,2048)
    for name,depth,columns in (('conv-in',2048,6144),('projection',2048,2048),('dense-down',11776,2048))
):
    p = dict(r=rows,k=depth,o=columns,optimized=True,ggml=True,type=2)
    spec = dict(kind='linear',parameters=p,contract='reference',phase='prefill')
    capabilities = dict(provider='cuda',target='sm_89')
    for bm,bn,bk,threads,stages in choices:
        axes = dict(block_m=bm,block_n=bn,block_k=bk,threads=threads,stages=stages)
        identifier = name + '-' + '-'.join(map(str,(bm,bn,bk,threads,stages)))
        path = root/(identifier+'.py')
        implementation,schedule = definition(spec,axes,capabilities,preferred='linear.cuda.packed:prefill')
        path.write_text(selected_source(spec,schedule,capabilities,implementation).render())
        if not path.with_suffix('.tbin').exists():
            build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89')
        records.append(dict(name=identifier,case=name,parameters=p,schedule=axes))
        write(root/'compilation.json',records)
        print('BUILT',identifier,flush=True)

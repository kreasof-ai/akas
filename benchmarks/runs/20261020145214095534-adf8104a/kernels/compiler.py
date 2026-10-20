from pathlib import Path
from akas.compilation.emit import build_artifact
from akas.compilation.selection import source
root=Path('build/lfm-moe/codec-linear-production');root.mkdir(exist_ok=True)
for depth,columns,paired in ((2048,6144,False),(11776,2048,False),(2048,11776,True)):
 for bm,bn,bk,threads in ((64,64,64,128),(64,128,128,256)):
  kind='ffn' if paired else 'linear';p=dict(r=4096,k=depth,o=columns,type=2,ggml=True,block_m=bm,block_n=bn,block_k=bk,threads=threads)
  path=root/f'{kind}-{depth}-{columns}-{bm}-{bn}-{bk}-{threads}.py';path.write_text(source(kind,p,contract='reference',provider='cuda',target='sm_89',preferred='linear.cuda.packed:prefill').render())
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)

from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/router-columns');root.mkdir(exist_ok=True)
for rows,depth,columns in ((32,2048,64),(4096,2048,64),(3,128,7)):
 for micro in ((2,4,8) if columns==64 else (4,)):
  p=dict(r=rows,k=depth,o=columns,threads=128,columns_per_warp=micro)
  path=root/f'{rows}-{depth}-{columns}-{micro}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom router_columns import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n')
  build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)

from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/planar-q4')
for inverse in (False,True):
 path=root/('unpack.py' if inverse else 'pack.py');path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom planar_transfer import factory\ndef tensor_export():\n return {"kernel":factory('+str(64*1536*2048//32*18)+','+str(inverse)+')}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)

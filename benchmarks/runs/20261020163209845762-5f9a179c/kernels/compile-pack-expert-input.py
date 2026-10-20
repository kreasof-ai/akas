from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/packed-expert-activation');root.mkdir(exist_ok=True)
for name,factory in (('offsets','offsets_factory'),('gather','gather_factory')):
 p=dict(r=4096,top=4,experts=64,k=2048)
 path=root/f'{name}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom pack_expert_input import '+factory+'\ndef tensor_export():\n return {"kernel":'+factory+'('+repr(p)+')}\n');build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)

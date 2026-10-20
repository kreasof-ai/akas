from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/half2-decode');root.mkdir(exist_ok=True)
for depth,columns,routed in ((2048,1536,False),(1536,2048,True),(2048,7,False)):
 for threads in (128,256):
  p=dict(r=1,top=4,experts=64,k=depth,o=columns,input_dtype='float32',routed_input=routed,round_inputs=True,threads=threads,weight_parts=({'rows':64*columns,'format':'ggml','options':{'encoding':2}},))
  entry=root/f'{depth}-{columns}-{threads}.py'
  entry.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom half2_decode import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n')
  build_artifact(entry,entry.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',entry.name,flush=True)

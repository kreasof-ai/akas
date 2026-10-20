from pathlib import Path
from akas.compilation.emit import build_artifact
s=Path('build/lfm-moe/factored_attention.py').read_text().replace('prob[i,j] = scores[i,j] * value_scale[j,0]','prob[i,j] = scores[i,j] * value_scale[j,0] * 128')
s=s.replace('result[i, j] * (vs[slot, head, 0, 0] if native_half else 1)','result[i, j] * (vs[slot, head, 0, 0] if native_half else 0.0078125)')
Path('build/lfm-moe/factored_scaled_attention.py').write_text(s)
root=Path('build/lfm-moe/factored-scaled-attention');root.mkdir(exist_ok=True)
for bk in (64,128):
 p=dict(r=1,h=32,kh=8,d=64,capacity=131072,splits=64,chunk=1,kv_dtype='fp8',provider='cuda',optimized=True,key_rows=bk,stages=1)
 path=root/f'131072-{bk}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom factored_scaled_attention import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n return {"kernel":factory(KernelSpec("attention_partial",'+repr(p)+'),{})}\n')
 build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',path.name,flush=True)

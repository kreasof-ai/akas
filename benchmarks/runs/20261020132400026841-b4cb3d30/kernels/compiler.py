from pathlib import Path
from akas.compilation.emit import build_artifact
root=Path('build/lfm-moe/prefill-occupancy');root.mkdir(exist_ok=True)
s=Path('build/lfm-moe/grouped_prefill.py').read_text()
s=s.replace("slots=r//chunk;group=h//kh;qm=", "slots=r//chunk;actual_group=h//kh;group_queries=p.get('group_queries',True);group=actual_group if group_queries else 1;work_heads=kh if group_queries else h;threads=p.get('threads',128);qm=")
s=s.replace('T.ceildiv(chunk,qt),kh,slots,threads=128','T.ceildiv(chunk,qt),work_heads,slots,threads=threads')
s=s.replace('   length=T.min(', '   cache_head=head if group_queries else head//actual_group\n   length=T.min(')
for name in ('kc','vc','ks','vs','keys','values'):s=s.replace(name+'[slot,head,',name+'[slot,cache_head,')
s=s.replace('read(kc,ks,slot,head,','read(kc,ks,slot,cache_head,').replace('read(vc,vs,slot,head,','read(vc,vs,slot,cache_head,')
Path('build/lfm-moe/prefill_occupancy.py').write_text(s)
variants=(('g64t128',64,128,True),('g64t256',64,256,True),('g128t256',128,256,True),('h64t128',64,128,False))
for dtype in ('float16','fp8'):
 for name,qm,threads,grouped in variants:
  for small in (False,True):
   p=dict(r=33 if small else 4096,h=8 if small else 32,kh=2 if small else 8,d=64,capacity=4096 if small else 131072,chunk=33 if small else 4096,kv_dtype=dtype,unit_scales=dtype=='float16',query_rows=qm,key_rows=64,stages=1,pipeline_group=8,threads=threads,group_queries=grouped)
   path=root/f'{dtype}-{name}{"-small" if small else ""}.py';path.write_text('import sys\nsys.path.insert(0,"/home/sagemaker-user/akas/build/lfm-moe")\nfrom prefill_occupancy import factory\ndef tensor_export():\n return {"kernel":factory('+repr(p)+')}\n')
   build_artifact(path,path.with_suffix('.tbin'),provider='cuda',target='sm_89');print('BUILT',dtype,name,small,flush=True)

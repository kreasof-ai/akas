"""Isolate wider native MMA tiles after moving FP8 conversion outside attention."""
import json
import time
import traceback
from pathlib import Path
from akas.compilation.emit import build_artifact

root=Path("build/lfm-moe/attention-tiles")
root.mkdir(exist_ok=True)
geometries=[(32,64,128),(64,64,128),(64,64,256),(128,64,128),(128,64,256),(64,128,256),(64,128,512)]
cases=[]
for label,slots,chunk,h,kh,cap in (("small",2,33,8,2,4096),("full",1,4096,32,8,131072)):
    for qm,bk,threads in ([(32,64,128),(128,64,256),(64,128,256)] if label=="small" else geometries):
        p=dict(r=slots*chunk,h=h,kh=kh,d=64,capacity=cap,chunk=chunk,kv_dtype="float16",unit_scales=True,
               query_rows=qm,key_rows=bk,threads=threads,stages=1,async_loads=True)
        name=f"{label}-{qm}-{bk}-{threads}"
        source=root/f"{name}.py"
        source.write_text("import sys\nsys.path.insert(0,'"+str(root.parent.absolute())+"')\nfrom attention_wide import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n    return {'kernel':factory(KernelSpec('attention',"+repr(p)+"),{})}\n")
        started=time.monotonic()
        try:
            source.with_suffix(".tbin").unlink(missing_ok=True)
            build_artifact(source,source.with_suffix(".tbin"),provider="cuda",target="sm_89")
            check=dict(name=name,label=label,parameters=p,passed=True,seconds=time.monotonic()-started)
            print("BUILT",name,check["seconds"],flush=True)
        except Exception:
            check=dict(name=name,label=label,parameters=p,passed=False,seconds=time.monotonic()-started,traceback=traceback.format_exc())
            print("REJECTED",name,check["traceback"],flush=True)
        cases.append(check)
        (root/"compilation.json").write_text(json.dumps(cases,indent=2)+"\n")

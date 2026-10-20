"""Isolate wider native MMA tiles after moving FP8 conversion outside attention."""
import json
import time
import traceback
from pathlib import Path
from akas.compilation.emit import build_artifact

root=Path("build/lfm-moe/attention-rows-large")
root.mkdir(exist_ok=True)
geometries=[(64,64,128,1),(64,64,128,2),(128,64,256,2),(256,64,512,1),(256,64,512,2),(64,32,128,1),(64,32,128,2)]
cases=[]
for label,slots,chunk,h,kh,cap in (("small",2,33,8,2,4096),("full",1,4096,32,8,131072)):
    for qm,bk,threads,stages in ([(32,64,128,1),(256,64,512,1),(64,32,128,2)] if label=="small" else geometries):
        p=dict(r=slots*chunk,h=h,kh=kh,d=64,capacity=cap,chunk=chunk,kv_dtype="float16",unit_scales=True,
               query_rows=qm,key_rows=bk,threads=threads,stages=stages,async_loads=True)
        name=f"{label}-{qm}-{bk}-{threads}" if label=="small" and qm==32 else f"{label}-{qm}-{bk}-{threads}-{stages}"
        source=root/f"{name}.py"
        source.write_text("import sys\nsys.path.insert(0,'"+str(root.parent.absolute())+"')\nfrom attention_rows_large import factory\nfrom akas.kernels.definition import KernelSpec\ndef tensor_export():\n    return {'kernel':factory(KernelSpec('attention',"+repr(p)+"),{})}\n")
        started=time.monotonic()
        try:
            source.with_suffix(".tbin").unlink(missing_ok=True)
            build_artifact(source,source.with_suffix(".tbin"),provider="cuda",target="sm_89")
            from tensor.artifacts.format import read_artifact
            metadata,_=read_artifact(source.with_suffix(".tbin"))
            shared=metadata["launch"].get("shared_memory_bytes",0)
            if shared>101376:raise ValueError(f"L4 shared memory limit exceeded: {shared}")
            check=dict(name=name,label=label,parameters=p,passed=True,seconds=time.monotonic()-started,shared_memory_bytes=shared)
            print("BUILT",name,check["seconds"],flush=True)
        except Exception:
            check=dict(name=name,label=label,parameters=p,passed=False,seconds=time.monotonic()-started,traceback=traceback.format_exc())
            print("REJECTED",name,check["traceback"],flush=True)
        cases.append(check)
        (root/"compilation.json").write_text(json.dumps(cases,indent=2)+"\n")

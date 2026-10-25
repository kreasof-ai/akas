"""Bounded paired decode geometry comparison; compilation only, no GPU timing."""
import json
import time
import traceback
from pathlib import Path
from akas.compilation.emit import build_artifact

root=Path("build/lfm-moe/paired-decode")
root.mkdir(exist_ok=True)
cases=[("small",dict(k=256,o=37,experts=6,top=4,threads=128,width=1,unroll=1),"factory")]
cases += [(f"full-{threads}-{width}-{unroll}",dict(k=2048,o=1536,experts=64,top=4,threads=threads,width=width,unroll=unroll),"factory")
          for threads,width,unroll in ((128,1,1),(128,1,2),(256,1,1),(128,2,1),(128,2,2),(256,2,1))]
cases += [("evict",dict(elements=128*1024*1024//4),"evict_factory")]
checks=[]
for name,p,factory in cases:
    if factory=="factory":
        p["weight_parts"]=({"rows":p["experts"]*p["o"],"format":"ggml","options":{"encoding":2}},)
    path=root/f"{name}.py"
    path.write_text("import sys\nsys.path.insert(0,'"+str(root.parent.absolute())+"')\nfrom paired_decode import "+factory+"\ndef tensor_export():\n    return {'kernel':"+factory+"("+repr(p)+")}\n")
    started=time.monotonic()
    try:
        path.with_suffix(".tbin").unlink(missing_ok=True)
        build_artifact(path,path.with_suffix(".tbin"),provider="cuda",target="sm_89")
        result=dict(name=name,parameters=p,passed=True,seconds=time.monotonic()-started)
        print("BUILT",name,result["seconds"],flush=True)
    except Exception:
        result=dict(name=name,parameters=p,passed=False,seconds=time.monotonic()-started,traceback=traceback.format_exc())
        print("REJECTED",name,result["traceback"],flush=True)
    checks.append(result)
    (root/"compilation.json").write_text(json.dumps(checks,indent=2)+"\n")

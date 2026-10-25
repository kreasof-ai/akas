"""Bounded full-fusion experiment, including small independent reference shape."""
import json
import time
import traceback
from pathlib import Path

from akas.compilation.emit import build_artifact

root = Path("build/lfm-moe/pattention-q4")
root.mkdir(exist_ok=True)
checks = []
shapes = [("small", 33, 2, 4, 256, 128, 128, 16, 32, 64, 64, 128)]
shapes += [(f"full-{bm}-{bn}-{bd}-{threads}", 4096, 4, 64, 2048, 1536, 2048, bm, bn, 64, bd, threads)
           for bm, bn, bd, threads in ((16, 32, 128, 128), (16, 32, 512, 128), (16, 32, 1024, 128), (32, 32, 1024, 256),
                                       (32, 64, 512, 256), (64, 64, 512, 512), (64, 32, 1024, 512))]
for name, rows, top, experts, depth, hidden, columns, bm, bn, bk, bd, threads in shapes:
    p = dict(r=rows, top=top, experts=experts, k=depth, hidden=hidden, o=columns,
             block_m=bm, block_n=bn, block_k=bk, block_d=bd, threads=threads,
             weight_parts=({"rows":experts*hidden,"format":"ggml","options":{"encoding":2}},),
             down_parts=({"rows":experts*columns,"format":"ggml","options":{"encoding":2}},))
    source = root / f"{name}.py"
    source.write_text("import sys\nsys.path.insert(0, '"+str(root.parent.absolute())+"')\nfrom pattention_q4_expert import factory\ndef tensor_export():\n    return {'kernel':factory("+repr(p)+")}\n")
    started = time.monotonic()
    try:
        source.with_suffix(".tbin").unlink(missing_ok=True)
        build_artifact(source, source.with_suffix(".tbin"), provider="cuda", target="sm_89")
        check = dict(name=name, parameters=p, passed=True, seconds=time.monotonic()-started)
        print("BUILT", name, check["seconds"], flush=True)
    except Exception:
        check = dict(name=name, parameters=p, passed=False, seconds=time.monotonic()-started, traceback=traceback.format_exc())
        print("REJECTED", name, check["traceback"], flush=True)
    checks.append(check)
    (root / "compilation.json").write_text(json.dumps(checks, indent=2)+"\n")

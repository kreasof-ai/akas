"""Run any declarative Qwen research preset on the shared H200 volume.

Example: modal run benchmarks/qwen35/modal_execute.py --checkpoint
/cache/models/<revision> --preset modal_hopper_long --out build/hopper-report.json
"""

import modal
from benchmarks.qwen35.modal_environment import image, volume

app = modal.App("akas-program-research")


@app.function(image=image, gpu="H200", cpu=16, memory=98304, timeout=3600,
              volumes={"/cache": volume}, scaledown_window=2)
def execute(checkpoint, preset, workload=None, prepared=None, window=None):
    import hashlib
    from importlib import import_module
    import json
    import os
    from pathlib import Path
    from akas.benchmarking.research import prepare, run_case, run_workload

    if not preset.replace("_", "").isalnum():
        raise ValueError("preset must name a research module")
    os.chdir("/workspace")
    volume.reload()
    case = import_module("benchmarks.qwen35." + preset)
    settings = dict(case.PRESET)
    identity = hashlib.sha256(json.dumps(dict(checkpoint=checkpoint, settings=settings), sort_keys=True).encode()).hexdigest()[:16]
    if prepared:
        ready = json.loads(Path(prepared).read_text())
    else:
        ready = prepare(case.DEFINITION, checkpoint, Path("/cache/programs") / identity,
                        **{name: settings[name] for name in ("batches", "chunk", "windows", "context", "target", "model_options", "policy")})
    if workload:
        report = run_workload(ready, workload, window=window)
    else:
        report = run_case(ready, batch=settings["batch"], prompt=settings["prompt"], generated=settings["generated"])
    result = dict(prepared=ready, report=report)
    (Path(ready["program"]) / "remote-report.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(checkpoint: str, preset: str = "modal_h200", workload: str = "", prepared: str = "",
         window: int = 0, out: str = "build/research-h200.json"):
    import json
    from pathlib import Path
    report = execute.remote(checkpoint, preset, workload or None, prepared or None, window or None)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")

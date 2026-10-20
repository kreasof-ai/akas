"""Observe actual routed expert loads before the shared count buffers are reused."""
import json
import traceback
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.history import Recorder, write
from akas.benchmarking.session import Session
from akas.execution.plan import call_bindings
from akas.optimization.evaluation import prompts

root = Path("build/lfm-moe/routing-counts")
root.mkdir(exist_ok=True)
parent = "20261020152431734974-3777f95e"
reference = json.loads((Path("benchmarks/runs")/parent/"report.json").read_text())
meta = dict(runtime="akas", purpose="kernel", protocol="routing-distribution-v1", provider="cuda", target="sm_89",
            device=dict(name="NVIDIA L4",provider="cuda"), model="akas.models.lfm2.model:LFM2MoE",
            checkpoint="build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf", checkpoint_sha256="d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5",
            weight_format="q4_0", kv_dtype="fp8", mode="ar", shape_timeout_seconds=120, reference_run_id=parent,
            qualification="Untimed per-layer expert routing counts on the qualified full model. Counts copied immediately after routing before aliased workspace reuse. Synchronization perturbs performance; no latency or throughput claim. Prefill first greedy token must match the qualified FP8 reference.")
rec = Recorder("benchmarks",meta,kind="diagnostic",change="Measure per-layer real expert load imbalance and useful row fractions before selecting fusion geometry",parent=parent)
report = dict(schema="akas.benchmark.v1",metadata=meta,results=[])
passed = False
try:
    with tensor.Device(provider="cuda") as device:
        with LFM2MoE(meta["checkpoint"],"build/lfm-moe/codec-model/cuda/program",device,graphs=True) as model:
            for prefix, chunk in ((32,32),(8192,4096)):
                captured = []
                with ExitStack() as stack:
                    for phase in ("prefill", "prefill_state"):
                        try:
                            forward = model.batches.select(1,chunk,phase=phase).forward
                        except (ValueError,KeyError):
                            continue
                        original = forward.run
                        stack.callback(lambda f=forward,run=original:setattr(f,"run",run))
                        def observe(f=forward,phase_name=phase):
                            invocation = len({(x["phase"],x["invocation"]) for x in captured})
                            for entry, call in zip(f.plan,f.definition.calls):
                                device._launch(entry[0],entry[1])
                                if call.kind == "quantized_routes":
                                    counts = call_bindings(entry[0],entry[1])["counts"].to_numpy().reshape(-1)
                                    if np.any(counts < 0):
                                        raise ValueError("Negative expert count")
                                    useful = int(counts.sum())
                                    padding = {}
                                    for bm in (16,32,64,128):
                                        allocated = int((((counts+bm-1)//bm)*bm).sum())
                                        padding[str(bm)] = dict(allocated_rows=allocated,useful_fraction=useful/allocated if allocated else 1.0)
                                    captured.append(dict(phase=phase_name,invocation=invocation,label=entry[2],rows=call.parameters.get("r"),counts=counts.tolist(),
                                                         total_routes=useful,active_experts=int(np.count_nonzero(counts)),min=int(counts.min()),max=int(counts.max()),
                                                         quantiles=np.quantile(counts,[0,.25,.5,.75,1]).tolist(),padding=padding))
                        forward.run = observe
                    with Session(model,batch=1,chunk=chunk,timeout=120) as session:
                        session.reset()
                        actual = list(session.prefill(prompts(model.config.vocab,1,prefix)))
                    expected = next(r for r in reference["results"] if r["case"]["prompt"]==prefix)["output_ids"][0][0]
                    if actual != [expected]:
                        raise ValueError(f"Routing diagnostic changed the first token at {prefix}: {actual} vs {expected}")
                if not captured:
                    raise ValueError("No expert routing calls observed")
                meta.setdefault("profiles",[]).append(dict(prefix=prefix,chunk=chunk,calls=captured,first_token_matches=True))
                summary = dict(prefix=prefix,observations=len(captured),active_experts_min=min(x["active_experts"] for x in captured),
                               active_experts_max=max(x["active_experts"] for x in captured),max_expert_rows=max(x["max"] for x in captured),
                               useful_fractions={str(bm):float(np.mean([x["padding"][str(bm)]["useful_fraction"] for x in captured])) for bm in (16,32,64,128)})
                meta.setdefault("summaries",[]).append(summary)
                print("ROUTING",json.dumps(summary),flush=True)
    passed = True
except BaseException:
    meta["traceback"] = traceback.format_exc()
    print(meta["traceback"],flush=True)
finally:
    rec.artifact(__file__,"diagnostic/profile-routing-counts.py")
    write(root/"report.json",report)
    rec.update(report,final=True,status="passed" if passed else "failed")
    print("RETAINED",rec.id,passed,flush=True)
if not passed:
    raise SystemExit(1)

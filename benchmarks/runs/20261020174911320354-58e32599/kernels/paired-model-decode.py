"""Whole-model A/B/A decode over identical state, with the paired expert kernel."""
import json
import statistics
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

import tensor
from akas.models.lfm2.model import LFM2MoE
from akas.benchmarking.history import Recorder,write
from akas.benchmarking.session import Session
from akas.execution.plan import ExecutionPlan,call_bindings
from akas.optimization.evaluation import prompts

kernel_root=Path("build/lfm-moe/paired-decode")
kernel_report=json.loads((kernel_root/"report.json").read_text())
checks=kernel_report["metadata"]["checks"]
baseline=next(x for x in checks if x.get("cache")=="cold" and x.get("geometry")=="three-kernel-baseline")
candidates=[x for x in checks if x.get("cache")=="cold" and x.get("bitwise") and x.get("geometry","").startswith("full-")]
if not candidates:raise ValueError("No bitwise paired decode candidate qualified")
winner=min(candidates,key=lambda x:x["timing"]["median_seconds"])
if winner["timing"]["median_seconds"]>=baseline["timing"]["median_seconds"]*.98:
    print("SKIP_MODEL_NO_COLD_KERNEL_GAIN",flush=True)
    raise SystemExit(0)
parent="20261020152431734974-3777f95e"
reference=json.loads((Path("benchmarks/runs")/parent/"report.json").read_text())
program=Path("build/lfm-moe/codec-model/cuda/program")
integrated="20261020172334890603-b8fd2763"
qualification=json.loads((Path("benchmarks/runs")/integrated/"run.json").read_text())
if qualification["status"]=="passed":
    program=Path("build/lfm-moe/staged-model/cuda/program")
    qualified_program_run_id=integrated
else:
    qualified_program_run_id=parent
root=Path("build/lfm-moe/paired-model-decode")
root.mkdir(exist_ok=True)
meta=dict(runtime="akas",purpose="kernel",protocol="snapshot-decode-v1",provider="cuda",target="sm_89",device=dict(name="NVIDIA L4",provider="cuda"),
          model="akas.models.lfm2.model:LFM2MoE",checkpoint="build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf",
          checkpoint_sha256="d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5",weight_format="q4_0",kv_dtype="fp8",mode="ar",output_tokens=128,shape_timeout_seconds=120,
          qualified_program_run_id=qualified_program_run_id,reference_http_run_id=parent,geometry=winner["geometry"],parameters=winner["parameters"],
          qualification="Whole-model streamed weights, exact original/paired/original state snapshots. Pure AR, FP8 KV; same Q4 coefficients, F16 operands and F32 accumulators. All 128 greedy IDs must match the qualified FP8 reference before timing; 1 warmup and 3 measured 127-step repeats. Prefill prompts use baseline seed=17+B*4099+P. Decode-only prototype; no prefill speed or HTTP claim.")
rec=Recorder("benchmarks",meta,kind="algorithm",change="Qualify paired expert decode fusion under full-model streaming weights and identical AR state snapshots",parent=qualified_program_run_id)
report=dict(schema="akas.benchmark.v1",metadata=meta,results=[])
passed=False
try:
    with tensor.Device(provider="cuda") as device:
        with LFM2MoE(meta["checkpoint"],program,device,graphs=True) as model,ExitStack() as stack:
            forward=model.batches.select(1,1).forward
            original=forward.graph
            module=device.load(kernel_root/(winner["geometry"]+".tbin"));stack.callback(module.release)
            plan=ExecutionPlan(device,labeled=True)
            entries=list(zip(forward.plan,forward.definition.calls))
            index=0;replacements=0
            while index<len(entries):
                entry,call=entries[index]
                if call.kind=="quantized_experts" and not call.parameters.get("routed_input"):
                    up_entry,up_call=entries[index+1]
                    act_entry,act_call=entries[index+2]
                    assert up_call.kind=="quantized_experts" and act_call.kind=="swiglu_experts"
                    assert (call.parameters["k"],call.parameters["o"],call.parameters["experts"],call.parameters["top"])==(2048,1536,64,4)
                    gate=call_bindings(entry[0],entry[1]);up=call_bindings(up_entry[0],up_entry[1]);act=call_bindings(act_entry[0],act_entry[1])
                    bindings=dict(gate,w2=up["w"],out=act["out"])
                    plan.bind(module,tuple(bindings[a["name"]] for a in module.manifest["arguments"]),label=entry[2])
                    replacements+=1;index+=3
                else:
                    plan.append(entry);index+=1
            assert replacements==38
            candidate=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+(module,))
            stack.callback(candidate.close)
            stack.callback(lambda:setattr(forward,"graph",original))
            meta["replaced_expert_blocks"]=replacements
            print("OVERLAY_READY",replacements,winner["geometry"],flush=True)
            for prefix in (32,8192,130944):
                print("PREFILL_SNAPSHOT",prefix,flush=True)
                with Session(model,batch=1,chunk=4096,timeout=120) as session,ExitStack() as snapshots:
                    forward.graph=original;session.reset()
                    first=list(session.prefill(prompts(model.config.vocab,1,prefix)))
                    positions=model.positions.copy()
                    states={}
                    for name,buffer in model.batches.states.items():
                        clone=device.empty(buffer.shape,str(buffer.dtype));clone.copy_from(buffer)
                        states[name]=clone;snapshots.callback(clone.release)
                    device.synchronize()
                    meta["snapshot_gpu_bytes"]=sum(b.nbytes for b in states.values())
                    def restore():
                        for name,clone in states.items():model.batches.states[name].copy_from(clone)
                        model.positions[:]=positions;session.tokens=list(first);session.feedback_ready=False;session.deadline=time.perf_counter()+120
                        device.synchronize()
                    expected=next(r for r in reference["results"] if r["case"]["prompt"]==prefix)["output_ids"][0]
                    for label in ("original-before","paired","original-after"):
                        forward.graph=candidate if label=="paired" else original
                        restore();output=list(first)
                        for _ in range(127):output.extend(session.decode(1))
                        if output!=expected:raise ValueError(f"{label} changes the 128-token FP8 sequence at {prefix}")
                        samples=[]
                        for iteration in range(4):
                            restore();start=time.perf_counter();session.decode(127,read=False);device.synchronize();elapsed=time.perf_counter()-start
                            if iteration:samples.append(127/elapsed)
                        result=dict(prefix=prefix,variant=label,decode_tokens_per_second=statistics.median(samples),samples=samples,greedy_identical=True)
                        meta.setdefault("decode_comparisons",[]).append(result)
                        write(root/"report.json",report)
                        print("DECODE_RESULT",json.dumps(result),flush=True)
            passed=True
except BaseException:
    meta["traceback"]=traceback.format_exc();print(meta["traceback"],flush=True)
finally:
    rec.program(program)
    for path in (Path(__file__),kernel_root.parent/"paired_decode.py",kernel_root.parent/"compile-paired-decode.py",kernel_root.parent/"validate-paired-decode.py",kernel_root/"report.json"):
        rec.artifact(path,"kernels/"+path.name)
    rec.artifact(kernel_root/(winner["geometry"]+".tbin"),"overlay/paired.tbin")
    write(root/"report.json",report)
    rec.update(report,final=True,status="passed" if passed else "failed")
    print("RETAINED",rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

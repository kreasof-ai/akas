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

kernel_root=Path("build/lfm-moe/paired-planar-decode")
raw_kernel_root=Path("build/lfm-moe/paired-decode")
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
root=Path("build/lfm-moe/paired-planar-model")
root.mkdir(exist_ok=True)
meta=dict(runtime="akas",purpose="kernel",protocol="snapshot-decode-v1",provider="cuda",target="sm_89",device=dict(name="NVIDIA L4",provider="cuda"),
          model="akas.models.lfm2.model:LFM2MoE",checkpoint="build/checkpoints/lfm2.5-24b-a2b/q4_0.gguf",
          checkpoint_sha256="d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5",weight_format="q4_0",kv_dtype="fp8",mode="ar",output_tokens=128,shape_timeout_seconds=120,
          qualified_program_run_id=qualified_program_run_id,reference_http_run_id=parent,geometry=winner["geometry"],parameters=winner["parameters"],
          qualification="Whole-model streamed weights, exact original/paired-raw/paired-planar/original state snapshots. Lossless Q4 byte permutation includes all 114 expert matrices; no change to any scale or quant bit. Pure AR, FP8 KV; same Q4 coefficients, F16 operands and F32 accumulators. All 128 greedy IDs must match the qualified FP8 reference before timing; 1 warmup and 3 measured 127-step repeats. Prefill prompts use baseline seed=17+B*4099+P. Decode-only prototype; no prefill speed or HTTP claim.")
rec=Recorder("benchmarks",meta,kind="algorithm",change="Combine paired expert decode fusion with a lossless aligned Q4 layout under full-model streaming weights",parent=qualified_program_run_id)
report=dict(schema="akas.benchmark.v1",metadata=meta,results=[])
passed=False
raw_winner=None
try:
    with tensor.Device(provider="cuda") as device:
        with LFM2MoE(meta["checkpoint"],program,device,graphs=True) as model,ExitStack() as stack:
            forward=model.batches.select(1,1).forward
            original=forward.graph
            module=device.load(kernel_root/(winner["geometry"]+".tbin"));stack.callback(module.release)
            raw_report=json.loads((raw_kernel_root/"report.json").read_text())
            raw_winner=min((x for x in raw_report["metadata"]["checks"] if x.get("cache")=="cold" and x.get("bitwise") and x.get("geometry","").startswith("full-")),key=lambda x:x["timing"]["median_seconds"])
            raw_module=device.load(raw_kernel_root/(raw_winner["geometry"]+".tbin"));stack.callback(raw_module.release)
            planar_root=Path("build/lfm-moe/planar-q4")
            down_module=device.load(planar_root/"1-1536-2048.tbin");stack.callback(down_module.release)
            entries=list(zip(forward.plan,forward.definition.calls))
            variants={}
            for label,pair_module in (("paired-raw",raw_module),("paired-planar",module)):
                plan=ExecutionPlan(device,labeled=True);index=0;replacements=0
                while index<len(entries):
                    entry,call=entries[index]
                    if call.kind=="quantized_experts" and not call.parameters.get("routed_input"):
                        up_entry,up_call=entries[index+1];act_entry,act_call=entries[index+2]
                        assert up_call.kind=="quantized_experts" and act_call.kind=="swiglu_experts"
                        gate=call_bindings(entry[0],entry[1]);up=call_bindings(up_entry[0],up_entry[1]);act=call_bindings(act_entry[0],act_entry[1])
                        bindings=dict(gate,w2=up["w"],out=act["out"])
                        plan.bind(pair_module,tuple(bindings[a["name"]] for a in pair_module.manifest["arguments"]),label=entry[2])
                        replacements+=1;index+=3
                    elif label=="paired-planar" and call.kind=="quantized_experts":
                        bindings=call_bindings(entry[0],entry[1])
                        plan.bind(down_module,tuple(bindings[a["name"]] for a in down_module.manifest["arguments"]),label=entry[2]);index+=1
                    else:plan.append(entry);index+=1
                assert replacements==38
                variants[label]=plan.capture(resources=tuple(forward.buffers.values())+tuple(forward.kernels.values())+(module,raw_module,down_module))
                stack.callback(variants[label].close)
            stack.callback(lambda:setattr(forward,"graph",original))
            pack=device.load(planar_root/"pack.tbin");unpack=device.load(planar_root/"unpack.tbin")
            stack.callback(pack.release);stack.callback(unpack.release)
            weight_names=sorted({call.arguments[1] for call in forward.definition.calls if call.kind=="quantized_experts"})
            weights=[model.weights[name] for name in weight_names]
            count=weights[0].nbytes
            assert len(weights)==114 and all(w.nbytes==count and str(w.dtype)=="uint8" for w in weights)
            temporary=device.empty((count,),"uint8");stack.callback(temporary.release)
            import numpy as np
            original_bytes=weights[0].to_numpy();blocks=original_bytes.reshape(-1,18)
            expected_layout=np.concatenate((blocks[:,2:].reshape(-1),blocks[:,:2].reshape(-1)))
            pack.launch(weights[0],temporary);np.testing.assert_array_equal(temporary.to_numpy(),expected_layout)
            del original_bytes,blocks,expected_layout
            meta["weight_layout"]=dict(tensors=114,resident_payload_change_bytes=0,temporary_bytes=count,transforms_outside_timing=True)
            meta["raw_geometry"]=raw_winner["geometry"]
            def transform(kernel):
                start=time.perf_counter()
                for weight in weights:kernel.launch(weight,temporary);weight.copy_from(temporary)
                device.synchronize();return time.perf_counter()-start
            print("OVERLAYS_READY",winner["geometry"],raw_winner["geometry"],flush=True)
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
                    for label in ("original-before","paired-raw","paired-planar","original-after"):
                        if label=="paired-planar":meta.setdefault("transforms",[]).append(dict(prefix=prefix,direction="pack",seconds=transform(pack)))
                        if label=="original-after":meta.setdefault("transforms",[]).append(dict(prefix=prefix,direction="unpack",seconds=transform(unpack)))
                        forward.graph=variants.get(label,original)
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
    for path in (Path(__file__),kernel_root.parent/"paired_decode_planar.py",kernel_root.parent/"compile-paired-planar-decode.py",kernel_root.parent/"validate-paired-planar-decode.py",kernel_root.parent/"planar_q4.py",kernel_root.parent/"planar_decode.py",kernel_root.parent/"planar_transfer.py",kernel_root/"report.json"):
        rec.artifact(path,"kernels/"+path.name)
    rec.artifact(kernel_root/(winner["geometry"]+".tbin"),"overlay/paired.tbin")
    if raw_winner is not None:rec.artifact(raw_kernel_root/(raw_winner["geometry"]+".tbin"),"overlay/paired-raw.tbin")
    for name in ("pack","unpack","1-1536-2048"):rec.artifact(Path("build/lfm-moe/planar-q4")/(name+".tbin"),"overlay/"+name+".tbin")
    write(root/"report.json",report)
    rec.update(report,final=True,status="passed" if passed else "failed")
    print("RETAINED",rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

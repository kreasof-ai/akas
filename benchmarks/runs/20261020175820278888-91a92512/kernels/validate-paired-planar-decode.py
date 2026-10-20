"""Qualify fusion, then compare complete gate/up/activation with warm/cold L2."""
import ctypes as c
import json
import resource
import signal
import statistics
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from tensor.providers.cuda.graph import CudaGraph
from akas.artifacts.read import read_program,artifact_path
from akas.benchmarking.history import Recorder,write
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.memory.transfer import write as transfer
from akas.quantization.weights.ggml.validation import sample_weight,decode_rows

root=Path("build/lfm-moe/paired-planar-decode")
program=Path("build/lfm-moe/codec-model/cuda/program")
meta=dict(runtime="akas",purpose="kernel",protocol="kernel-latency-v1",provider="cuda",target="sm_89",device=dict(name="NVIDIA L4",provider="cuda"),
          weight_format="q4_0",operand_dtype="float16",accumulator_dtype="float32",shape_timeout_seconds=120,
          qualification="Gate/up/SwiGLU decode fusion over the existing codec dot contract. Original Q4 weights; no activation quantization or pruning. Independent small CPU reference, current-AOT full-size comparison, inactive experts and unchanged inputs. Timing compares complete three-kernel baseline with fused kernel. Cold samples first overwrite 128 MiB of unrelated GPU memory outside the timed region; L4 L2 is 48 MiB. CUDA events bracket only the expert sequence, with one graph invocation per sample. This is a kernel result, not serving throughput.")
meta["storage_layout"]="Lossless separation of unchanged Q4 quant bytes and scale bits; same resident byte count, conversion outside timed work"
rec=Recorder("benchmarks",meta,kind="algorithm",change="Fuse decode expert gate/up/SwiGLU without recomputing projections; distinguish L2-resident and streamed-weight timings")
report=dict(schema="akas.benchmark.v1",metadata=meta,results=[])
checks=[]
passed=False

@contextmanager
def deadline():
    def expired(signum,frame):raise TimeoutError("Kernel shape exceeded 120 seconds")
    previous=signal.signal(signal.SIGALRM,expired)
    signal.alarm(120)
    try:yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)

def args_for(kernel,buffers):return tuple(buffers[a["name"]] for a in kernel.manifest["arguments"])

def properties(device,kernel):
    api=device.driver.lib.cuFuncGetAttribute
    api.argtypes=[c.POINTER(c.c_int),c.c_int,c.c_void_p]
    api.restype=c.c_int
    result={}
    for name,attribute in (("max_threads",0),("static_shared_bytes",1),("local_bytes_per_thread",3),("registers_per_thread",4)):
        value=c.c_int()
        device.driver.call("cuFuncGetAttribute",c.byref(value),attribute,kernel.function)
        result[name]=value.value
    return result

def fixture(signature,p):
    arrays=sample_arguments(signature)
    for seed,name in ((31,"w"),(32,"w2")):
        sample_weight(arrays[name],2,seed=seed,decode=False)
        # Periodic validation bytes otherwise make experts identical when the
        # expert bank stride is a multiple of the fixture period.
        blocks=arrays[name].reshape(p["experts"],p["o"],p["k"]//32,18)
        scales=blocks[...,:2].view("float16")[...,0]
        for expert in range(p["experts"]):
            scales[expert]*=np.float16(.5+expert/max(1,p["experts"]-1))
    arrays["ids"][:]=[0,1,3,5] if p["experts"]==6 else [2,17,39,63]
    arrays["counts"].fill(0)
    arrays["counts"][arrays["ids"][0]]=1
    arrays["out"].fill(1.25)
    return arrays

def prepare_planar(device,buffers,arrays):
    for name in ("w","w2"):
        blocks=arrays[name].reshape(-1,18)
        packed=np.concatenate((blocks[:,2:].reshape(-1),blocks[:,:2].reshape(-1)))
        if packed.nbytes<1048576:
            restored=np.empty_like(blocks)
            restored[:,2:]=packed[:len(blocks)*16].reshape(-1,16)
            restored[:,:2]=packed[len(blocks)*16:].reshape(-1,2)
            np.testing.assert_array_equal(restored.reshape(-1),arrays[name])
        buffers["planar_"+name]=device.from_numpy(packed)

def cpu_reference(arrays,p):
    result=np.zeros_like(arrays["out"])
    x=arrays["x"][0].astype("float16").astype("float64")
    for rank,expert in enumerate(arrays["ids"][0]):
        if not arrays["counts"][expert]:continue
        start,stop=expert*p["o"],(expert+1)*p["o"]
        gate=decode_rows(arrays["w"],2,p["k"],start,stop).astype("float16").astype("float64")
        up=decode_rows(arrays["w2"],2,p["k"],start,stop).astype("float16").astype("float64")
        g=(gate@x).astype("float32")
        u=(up@x).astype("float32")
        result[0,rank]=g/(1+np.exp(-g))*u
    return result

def warm_timing(device,calls):
    plan=ExecutionPlan(device)
    for kernel,args in calls:plan.bind(kernel,args)
    with plan.capture(resources=tuple(v for kernel,args in calls for v in (kernel,*args))) as graph:
        with deadline():return measure(device,graph.launch,warmup=5,samples=9,repeats=10)

def cold_timing(device,calls,evict,eviction):
    events=[c.c_void_p(),c.c_void_p()]
    device.driver.lib.cuEventRecordWithFlags.argtypes=[c.c_void_p,c.c_void_p,c.c_uint]
    device.driver.lib.cuEventRecordWithFlags.restype=c.c_int
    for event in events:device.driver.call("cuEventCreate",c.byref(event),0)
    def enqueue():
        evict.launch(eviction)
        device.driver.call("cuEventRecordWithFlags",events[0],device.stream,1)
        for kernel,args in calls:kernel.launch(*args)
        device.driver.call("cuEventRecordWithFlags",events[1],device.stream,1)
    samples=[]
    try:
        resources=(evict,eviction,*tuple(v for kernel,args in calls for v in (kernel,*args)))
        with CudaGraph(device,enqueue,resources=resources) as graph,deadline():
            for iteration in range(11):
                graph.launch();device.synchronize()
                value=c.c_float()
                device.driver.call("cuEventElapsedTime",c.byref(value),*events)
                if iteration>=2:samples.append(value.value/1000)
        return dict(median_seconds=statistics.median(samples),samples=samples,warmup=2,repeats=1,sample_count=len(samples),eviction_bytes=eviction.nbytes)
    finally:
        for event in events:device.driver.call("cuEventDestroy_v2",event)

try:
    compiled=json.loads((root/"compilation.json").read_text())
    small=next(x for x in compiled if x["name"]=="small" and x["passed"])
    with tensor.Device(provider="cuda") as device:
        signature,_=read_artifact(root/"small.tbin")
        arrays=fixture(signature["arguments"],small["parameters"])
        buffers={name:device.from_numpy(value) for name,value in arrays.items()}
        prepare_planar(device,buffers,arrays)
        try:
            with device.load(root/"small.tbin") as kernel:
                for mode in ("all-active","one-inactive","all-inactive"):
                    if mode=="one-inactive":arrays["counts"][arrays["ids"][0,1]]=0
                    if mode=="all-inactive":arrays["counts"].fill(0)
                    transfer(device,buffers["counts"],arrays["counts"])
                    transfer(device,buffers["out"],arrays["out"])
                    expected=cpu_reference(arrays,small["parameters"])
                    kernel.launch(*args_for(kernel,dict(buffers,w=buffers["planar_w"],w2=buffers["planar_w2"])))
                    np.testing.assert_allclose(buffers["out"].to_numpy(),expected,rtol=.002,atol=.0002)
                    for name,value in arrays.items():
                        if name!="out":np.testing.assert_array_equal(buffers[name].to_numpy(),value)
                    checks.append(dict(shape="small",mode=mode,correct=True,independent_cpu=True,output_tail=True,unchanged_inputs=True))
                    print("CPU_PASS",mode,flush=True)
        finally:
            for buffer in buffers.values():buffer.release()
        del arrays,buffers,expected
        candidates=[x for x in compiled if x["passed"] and x["name"].startswith("full-")]
        if not candidates:raise ValueError("No full-size candidate compiled")
        signature,_=read_artifact(root/(candidates[0]["name"]+".tbin"))
        arrays=fixture(signature["arguments"],candidates[0]["parameters"])
        buffers={name:device.from_numpy(value) for name,value in arrays.items()}
        prepare_planar(device,buffers,arrays)
        top=read_program(program)
        manifest=json.loads((program/next(x["path"] for x in top["phases"] if x["phase"]=="decode" and x["chunk"]==1)).read_text())
        proj=artifact_path(program,next(k for k in manifest["kernels"].values() if k["kind"]=="quantized_experts" and k["parameters"]["k"]==2048))
        act=artifact_path(program,next(k for k in manifest["kernels"].values() if k["kind"]=="swiglu_experts"))
        try:
            with device.load(proj) as projection,device.load(act) as swiglu,device.load(root/"evict.tbin") as evict:
                buffers["gate"]=device.empty(buffers["out"].shape,"float32")
                buffers["up"]=device.empty(buffers["out"].shape,"float32")
                eviction=device.empty((128*1024*1024//4,),"int32")
                buffers["eviction"]=eviction
                evict.launch(eviction)
                actual=eviction.to_numpy();index=np.arange(0,len(actual),4093,dtype="int32")
                np.testing.assert_array_equal(actual[index],index);del actual,index
                gargs=args_for(projection,dict(buffers,out=buffers["gate"]))
                uargs=args_for(projection,dict(buffers,w=buffers["w2"],out=buffers["up"]))
                aargs=(buffers["gate"],buffers["up"],buffers["out"])
                baseline=[(projection,gargs),(projection,uargs),(swiglu,aargs)]
                for kernel,args in baseline:kernel.launch(*args)
                expected=buffers["out"].to_numpy()
                for mode,timing in (("warm",warm_timing(device,baseline)),("cold",cold_timing(device,baseline,evict,eviction))):
                    checks.append(dict(geometry="three-kernel-baseline",cache=mode,timing=timing,correct=True,properties=properties(device,projection)))
                    print("BASELINE",mode,timing["median_seconds"]*1e6,flush=True)
                for candidate in candidates:
                    with device.load(root/(candidate["name"]+".tbin")) as kernel:
                        transfer(device,buffers["out"],arrays["out"])
                        args=args_for(kernel,dict(buffers,w=buffers["planar_w"],w2=buffers["planar_w2"]))
                        kernel.launch(*args)
                        actual=buffers["out"].to_numpy()
                        np.testing.assert_allclose(actual,expected,rtol=.002,atol=.0002)
                        bitwise=bool(np.array_equal(actual.view("uint32"),expected.view("uint32")))
                        for mode,timing in (("warm",warm_timing(device,[(kernel,args)])),("cold",cold_timing(device,[(kernel,args)],evict,eviction))):
                            checks.append(dict(geometry=candidate["name"],parameters=candidate["parameters"],cache=mode,timing=timing,correct=True,bitwise=bitwise,
                                               max_absolute_error=float(np.max(np.abs(actual-expected))),launch=kernel.manifest["launch"],properties=properties(device,kernel)))
                            print("FUSED_PASS",candidate["name"],mode,timing["median_seconds"]*1e6,"bitwise",bitwise,flush=True)
                meta["selected_weight_bytes"]=2*4*1536*2048//32*18
                passed=True
        finally:
            for buffer in buffers.values():buffer.release()
except BaseException:
    meta["traceback"]=traceback.format_exc();print(meta["traceback"],flush=True)
finally:
    meta["checks"]=checks
    meta["cpu_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    for path in (Path(__file__),root.parent/"paired_decode_planar.py",root.parent/"compile-paired-planar-decode.py",root/"compilation.json",root.parent/"planar_q4.py"):
        if path.exists():rec.artifact(path,"kernels/"+path.name)
    write(root/"report.json",report)
    rec.update(report,final=True,status="passed" if passed else "failed")
    print("RETAINED",rec.id,passed,flush=True)
if not passed:raise SystemExit(1)

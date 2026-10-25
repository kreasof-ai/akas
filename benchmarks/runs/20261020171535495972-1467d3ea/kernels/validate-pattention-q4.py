"""Full expert fusion: bounded independent oracle and controlled sequence timing."""
import json
import resource
import signal
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.compiler.tuning import measure
from akas.artifacts.read import read_program, artifact_path
from akas.benchmarking.history import Recorder, write
from akas.execution.plan import ExecutionPlan
from akas.kernels.contracts import sample_arguments
from akas.memory.transfer import write as transfer
from akas.ops.moe.validation import tile_tables
from akas.quantization.weights.ggml.validation import sample_weight, decode_rows

root = Path("build/lfm-moe/pattention-q4")
program = Path("build/lfm-moe/codec-model/cuda/program")
top = read_program(program)
manifest = json.loads((program / next(r["path"] for r in top["phases"] if r["phase"] == "prefill" and r["chunk"] == 4096)).read_text())
meta = dict(runtime="akas", purpose="kernel", protocol="kernel-latency-v1", provider="cuda", target="sm_89",
            device=dict(name="NVIDIA L4", provider="cuda"), weight_format="q4_0", operand_dtype="float16", accumulator_dtype="float32",
            shape_timeout_seconds=120, warmup=3, samples=5, repeats=3,
            inspiration=dict(kernel="https://github.com/kreasof-ai/foveal-sparse-indexer/blob/main/foveal_indexer/pattention_triton.py",
                             revision="e3b081461049999b0a01465bdd20b0b2abe1d94e",
                             benchmarks="https://github.com/kreasof-ai/foveal-sparse-indexer/blob/main/BENCHMARKS.md"),
            qualification="Original Q4 payloads and original routing. Full gate/up/SwiGLU/down fusion, no pruning or lower precision. Small CPU F64 oracle using original F16 operand boundaries, large current-AOT comparison, fresh sentinel outputs, inactive routes and no-active expert cases. Balanced and skewed synthetic routes; not a serving result or model quality result.")
rec = Recorder("benchmarks", meta, kind="algorithm", change="Evaluate Pattention-style complete Q4 expert fusion against materialized and packed staged execution")
report = dict(schema="akas.benchmark.v1", metadata=meta, results=[])
checks = []
passed = False

@contextmanager
def deadline():
    def timeout(signum, frame):
        raise TimeoutError("Kernel shape exceeded 120 seconds")
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

def timed(device, calls):
    plan = ExecutionPlan(device)
    for kernel, args in calls:
        plan.bind(kernel, args)
    prepared = plan.capture(resources=tuple(item for kernel, args in calls for item in (kernel, *args)))
    try:
        with deadline():
            return measure(device, prepared.launch, warmup=3, samples=5, repeats=3)
    finally:
        prepared.close()

def routing(arrays, rows, top, experts, distribution):
    ids = ((np.arange(rows)[:, None] if distribution == "balanced" else np.zeros((rows, 1), dtype="int32")) + np.arange(top)[None, :]) % experts
    counts = np.zeros(experts, dtype="int32")
    routes = np.full((experts, rows), -1, dtype="int32")
    for expert in range(experts):
        owners, ranks = np.where(ids[:-1] == expert)
        counts[expert] = len(owners)
        routes[expert, :len(owners)] = owners * top + ranks
    arrays["counts"][:] = counts
    arrays["routes"][:] = routes
    arrays["tile_experts"][:], arrays["tile_rows"][:] = tile_tables(counts, arrays["tile_experts"].shape[1])
    return counts, routes

def args_for(kernel, buffers):
    return tuple(buffers[a["name"]] for a in kernel.manifest["arguments"])

def cpu_reference(arrays, rows, top, experts, depth, hidden, columns):
    expected = np.full((rows, top, columns), 1.25, dtype="float32")
    for expert in range(experts):
        route = arrays["routes"][expert, :arrays["counts"][expert]]
        if not len(route):
            continue
        x = arrays["x"][route // top].astype("float16").astype("float64")
        g = decode_rows(arrays["w"], 2, depth, expert * hidden, (expert + 1) * hidden).astype("float16").astype("float64")
        u = decode_rows(arrays["w2"], 2, depth, expert * hidden, (expert + 1) * hidden).astype("float16").astype("float64")
        down = decode_rows(arrays["wd"], 2, hidden, expert * columns, (expert + 1) * columns).astype("float16").astype("float64")
        gate, up = (x @ g.T).astype("float32"), (x @ u.T).astype("float32")
        activated = (gate / (1 + np.exp(-gate)) * up).astype("float16").astype("float64")
        expected[route // top, route % top] = activated @ down.T
    return expected

try:
    with tensor.Device(provider="cuda") as device:
        signature, _ = read_artifact(root / "small.tbin")
        arrays = sample_arguments(signature["arguments"])
        for seed, name in enumerate(("w", "w2", "wd"), 31):
            sample_weight(arrays[name], 2, seed=seed, decode=False)
        for distribution in ("balanced", "skewed"):
            routing(arrays, 33, 2, 4, distribution)
            arrays["out"].fill(1.25)
            expected = cpu_reference(arrays, 33, 2, 4, 256, 128, 128)
            buffers = {n:device.from_numpy(a) for n, a in arrays.items()}
            try:
                with device.load(root / "small.tbin") as kernel:
                    kernel.launch(*args_for(kernel, buffers))
                    actual = buffers["out"].to_numpy()
                    np.testing.assert_allclose(actual, expected, rtol=.002, atol=.0002)
                    for name, value in arrays.items():
                        if name != "out":
                            np.testing.assert_array_equal(buffers[name].to_numpy(), value)
                    transfer(device, buffers["counts"], np.zeros_like(arrays["counts"]))
                    transfer(device, buffers["out"], arrays["out"])
                    kernel.launch(*args_for(kernel, buffers))
                    np.testing.assert_array_equal(buffers["out"].to_numpy(), arrays["out"])
                checks.append(dict(shape="small", distribution=distribution, independent_cpu=True, unchanged_inputs=True, inactive_unchanged=True))
                print("CPU_PASS", distribution, flush=True)
            finally:
                for buffer in buffers.values():
                    buffer.release()
        del arrays, buffers, expected, actual
        candidates = [x for x in json.loads((root / "compilation.json").read_text()) if x["passed"] and x["name"].startswith("full-")]
        if not candidates:
            raise ValueError("No full-size candidates compiled")
        signature, _ = read_artifact(root / (candidates[0]["name"] + ".tbin"))
        arrays = sample_arguments(signature["arguments"])
        for seed, name in enumerate(("w", "w2", "wd"), 31):
            sample_weight(arrays[name], 2, seed=seed, decode=False)
        arrays["out"].fill(1.25)
        buffers = {n:device.from_numpy(a) for n, a in arrays.items()}
        try:
            with ExitStack() as stack:
                projection = stack.enter_context(device.load(artifact_path(program, next(k for k in manifest["kernels"].values() if k["kind"] == "quantized_experts" and k["parameters"]["k"] == 2048))))
                down = stack.enter_context(device.load(artifact_path(program, next(k for k in manifest["kernels"].values() if k["kind"] == "quantized_experts" and k["parameters"]["k"] == 1536))))
                swiglu = stack.enter_context(device.load(artifact_path(program, next(k for k in manifest["kernels"].values() if k["kind"] == "swiglu_experts"))))
                for name in ("gate", "up", "activated"):
                    buffers[name] = device.from_numpy(np.zeros((4096, 4, 1536), dtype="float32"))
                gate_args = args_for(projection, dict(buffers, out=buffers["gate"]))
                up_args = args_for(projection, dict(buffers, w=buffers["w2"], out=buffers["up"]))
                act_args = (buffers["gate"], buffers["up"], buffers["activated"])
                down_args = args_for(down, dict(buffers, x=buffers["activated"], w=buffers["wd"]))
                calls = [(projection, gate_args), (projection, up_args), (swiglu, act_args), (down, down_args)]
                packed_root = root.parent / "packed-expert-activation"
                prefix_kernel = stack.enter_context(device.load(packed_root/"offsets.tbin"))
                gather_kernel = stack.enter_context(device.load(packed_root/"gather.tbin"))
                fused_gate = stack.enter_context(device.load(packed_root/"128-64-64-256-1.tbin"))
                packed_down = stack.enter_context(device.load(root.parent/"packed-q4-expert/1536-128-128-128-256-1.tbin"))
                buffers["offsets"] = device.empty((65,), "int32")
                buffers["packed"] = device.empty((4096*4,2048), "float16")
                buffers["packed_activation"] = device.empty((4096*4,1536), "float16")
                packed_calls = [(prefix_kernel,args_for(prefix_kernel,buffers)), (gather_kernel,args_for(gather_kernel,buffers)),
                                (fused_gate,args_for(fused_gate,dict(buffers,x=buffers["packed"],out=buffers["packed_activation"]))),
                                (packed_down,args_for(packed_down,dict(buffers,x=buffers["packed_activation"],w=buffers["wd"])))]
                for distribution in ("balanced", "skewed"):
                    counts, routes = routing(arrays, 4096, 4, 64, distribution)
                    for name in ("counts", "routes", "tile_experts", "tile_rows"):
                        transfer(device, buffers[name], arrays[name])
                    transfer(device, buffers["out"], arrays["out"])
                    for kernel, args in calls:
                        kernel.launch(*args)
                    expected = buffers["out"].to_numpy()
                    timing = timed(device, calls)
                    checks.append(dict(shape="4096x4x2048x1536x2048", distribution=distribution, geometry="materialized-baseline", timing=timing, counts=counts.tolist()))
                    print("BASELINE", distribution, timing["median_seconds"]*1000, flush=True)
                    transfer(device, buffers["out"], arrays["out"])
                    for kernel, args in packed_calls:
                        kernel.launch(*args)
                    np.testing.assert_array_equal(buffers["out"].to_numpy(), expected)
                    timing = timed(device,packed_calls)
                    checks.append(dict(shape="4096x4x2048x1536x2048",distribution=distribution,geometry="packed-gate-up-fused-baseline",packing_included=True,timing=timing,bitwise=True))
                    print("PACKED_BASELINE",distribution,timing["median_seconds"]*1000,flush=True)
                    for candidate in candidates:
                        with device.load(root / (candidate["name"] + ".tbin")) as kernel:
                            transfer(device, buffers["out"], arrays["out"])
                            args = args_for(kernel, buffers)
                            with deadline():
                                kernel.launch(*args)
                                actual = buffers["out"].to_numpy()
                            np.testing.assert_allclose(actual, expected, rtol=.002, atol=.0002)
                            np.testing.assert_array_equal(actual[-1], arrays["out"][-1])
                            timing = timed(device, [(kernel, args)])
                            check = dict(shape="4096x4x2048x1536x2048", distribution=distribution, geometry=candidate["name"], parameters=candidate["parameters"], launch=kernel.manifest["launch"], correct=True,
                                         bitwise=bool(np.array_equal(actual, expected)), max_absolute_error=float(np.max(np.abs(actual-expected))), timing=timing)
                            checks.append(check)
                            print("FULL_PASS", distribution, candidate["name"], timing["median_seconds"]*1000, "bitwise", check["bitwise"], flush=True)
                    del expected, actual
        finally:
            for buffer in buffers.values():
                buffer.release()
    passed = True
except BaseException:
    meta["traceback"] = traceback.format_exc()
    print(meta["traceback"], flush=True)
finally:
    meta["checks"] = checks
    meta["cpu_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    for path in (Path(__file__), root.parent/"pattention_q4_expert.py", root.parent/"compile-pattention-q4.py", root/"compilation.json"):
        if path.exists():
            rec.artifact(path, "kernels/"+path.name)
    write(root/"report.json", report)
    rec.update(report, final=True, status="passed" if passed else "failed")
    print("RETAINED", rec.id, passed, flush=True)
if not passed:
    raise SystemExit(1)

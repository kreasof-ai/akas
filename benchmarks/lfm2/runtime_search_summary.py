"""Archive accepted runtime ablation, wider search and full-model replay."""

from akas.compilation.paths import source_path
from akas.compilation.paths import tensor_source

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from tensor.artifacts.format import read_artifact
from akas.artifacts.fingerprint import packed_hashes as implementation_hashes


def run(root, out):
    root = source_path(root)
    load = lambda p: json.loads((root / p).read_text())
    initial = load("runtime-comparison/report.json")
    final = load("comparison/report.json")
    search = load("gemv-extended/report.json")
    recheck = load("gemv-recheck/report.json")
    fusion = load("fusion-v3/report.json")
    if (
        initial["status"] != "passed"
        or final["status"] != "passed"
        or search["status"] != "finished"
        or recheck["status"] != "passed"
        or fusion["status"] != "finished"
    ):
        raise ValueError("requires completed, validated research")
    if any(
        bundle["implementation"] != implementation_hashes("webgpu")
        for bundle in final["bundles"].values()
    ):
        raise ValueError("implementation changed since final measurement")
    sources = {}
    paths = [
        "src/tensor/providers/webgpu.py",
        'src/akas/execution/bindings/packed/resources.py',
        *('src/akas/compilation/source.py', 'src/akas/ops/attention/webgpu/packed.py', 'src/akas/ops/linear/webgpu/packed.py', 'src/akas/ops/normalization/webgpu/packed.py', 'src/akas/ops/quantization/webgpu/packed.py', 'src/akas/ops/quantization/webgpu/storage.py', 'src/akas/ops/reductions/webgpu.py', 'src/akas/ops/sequence/webgpu/packed.py', 'src/akas/ops/vocabulary/webgpu/packed.py', 'src/akas/ops/definition.py'),
        "src/tensor/compiler/webgpu_lowering.py",
        "src/tensor/compiler/search.py",
        "src/tensor/compiler/webgpu/schedules/search_space.py",
        "tests/providers/test_webgpu.py",
        "tests/providers/test_webgpu_lowering.py",
        "tests/compiler/test_webgpu_decode_schedule.py",
    ]
    paths += [str(p) for p in source_path("benchmarks/lfm2").glob("runtime*.py")]
    paths += [
        "benchmarks/lfm2/decode_fusion_search.py",
        "benchmarks/lfm2/decode_kernel_search.py",
        "benchmarks/lfm2/decode_search_recheck.py",
        "benchmarks/lfm2/decode_search_compare.py",
        'src/akas/compilation/recipes/packed.py',
    ]
    for p in paths:
        sources[p] = dict(
            sha256=hashlib.sha256(source_path(p).read_bytes()).hexdigest(),
            text=source_path(p).read_text(),
        )
    snapshots = {
        p.name: dict(
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(), text=p.read_text()
        )
        for p in (root / "runtime-sources").glob("*.py")
    }
    for module, path in [
        ("akas.execution.bindings.packed.resources", "model.py"),
        ("akas.ops.definition", "webgpu_kernels.py"),
        (
            "tensor.providers.webgpu.adapter",
            "tensor.providers.webgpu.buffer",
            "tensor.providers.webgpu.device",
            "tensor.providers.webgpu.plan",
            "webgpu.py",
        ),
    ]:
        if (
            snapshots[path]["sha256"]
            != initial["bundles"]["searched"]["implementation"][module]
        ):
            raise ValueError("runtime-only source snapshot mismatch: " + module)
    shaders = {}
    for name, directory in [
        ("before", "runtime"),
        ("runtime", "runtime"),
        ("searched", "extended"),
    ]:
        shaders[name] = {}
        for key, record in final["bundles"][name]["kernels"].items():
            artifact = root / directory / record["artifact"]
            manifest, files = read_artifact(artifact)
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("bundle checksum changed")
            shaders[name][key] = dict(
                sha256=hashlib.sha256(files["kernel.wgsl"]).hexdigest(),
                text=files["kernel.wgsl"].decode(),
            )
    numerical = {
        name: dict(
            max_relative_rms=max(
                row[name]["relative_rms"] for row in final["validation"]
            ),
            min_cosine=min(row[name]["cosine"] for row in final["validation"]),
            matching_argmax=sum(
                row[name]["argmax"][0] == row[name]["argmax"][1]
                for row in final["validation"]
            ),
        )
        for name in ("tensor_before", "tensor_runtime", "tensor_searched", "llama_cpp")
    }
    bitwise = all(
        np.array_equal(
            np.load(root / "comparison" / f"{i}-tensor_before.npy", allow_pickle=False),
            np.load(
                root / "comparison" / f"{i}-tensor_runtime.npy", allow_pickle=False
            ),
        )
        for i in range(len(final["validation"]))
    )
    if not bitwise:
        raise ValueError("runtime ablation changed logits")
    report = dict(
        schema="tensor.lfm2-runtime-search.v1",
        status="passed",
        repository_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        runtime_ablation=initial,
        full_model=final,
        projection_search=search,
        projection_recheck=recheck,
        attention_fusion_search=fusion,
        profiles={
            name: load("profile-" + name + ".json") for name in ("before", "after")
        },
        plan_counts=load("plan-counts.json"),
        sources=sources,
        runtime_source_snapshots=snapshots,
        bundle_shaders=shaders,
        numerical_summary=numerical,
        runtime_logits_bitwise_equal=bitwise,
        test_logs={p.name: p.read_text() for p in root.glob("*tests.log")},
    )
    out = source_path(out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                numerical=numerical,
                benchmarks=[
                    {k: v for k, v in row.items() if k != "samples"}
                    for row in final["benchmarks"]
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.out)

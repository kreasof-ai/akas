"""Bounded single-H200 qualification and C8 replay using pinned native artifacts.

Run: modal run benchmarks/qwen35/modal_h200.py --out build/qwen35-h200
Weights and compiled bundles persist in a Modal Volume. CPU preparation precedes
the GPU function; only the native engine runs, with no baseline engine jobs.
"""

import os
from pathlib import Path
import modal
from akas.compilation.paths import tensor_source

ROOT = (
    Path(os.environ["TENSOR_WORKSPACE"])
    if "TENSOR_WORKSPACE" in os.environ
    else Path(__file__).resolve().parents[2]
)
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
VOLUME_NAME = "tensor-qwen35-h200"
WORKLOAD = Path(
    os.environ.get(
        "TENSOR_H200_WORKLOAD",
        str(ROOT / "docs/research/data/qwen35-native-h200/workload.json"),
    )
)
app = modal.App("tensor-qwen35-h200")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "numpy==1.26.4",
        "regex==2026.9.10",
        "tilelang==0.1.14",
        "apache-tvm-ffi==0.1.12",
        "huggingface_hub==0.36.0",
        "tokenizers==0.22.2",
        "aiohttp==3.14.3",
        "pytest==9.1.1",
    )
    .add_local_file(
        ROOT / "tools/bootstrap/bootstrap_nvrtc.py",
        "/tmp/bootstrap_nvrtc.py",
        copy=True,
    )
    .run_commands("python /tmp/bootstrap_nvrtc.py --out /opt/tensor-nvrtc")
    .env(
        {
            "PYTHONPATH": "/workspace:/workspace/src:/workspace/tensor-dependency",
            "TENSOR_WORKSPACE": "/workspace",
            "TENSOR_NVRTC_HOME": "/opt/tensor-nvrtc",
            "TENSOR_CACHE_DIR": "/cache/compiler-cache",
            "TENSOR_QWEN_TARGET": "sm_90",
            "OMP_NUM_THREADS": "4",
        }
    )
)


def with_workspace(base):
    return (
        base.add_local_dir(
            tensor_source(),
            "/workspace/tensor-dependency/tensor",
            ignore=["**/__pycache__/**"],
        )
        .add_local_dir(
            ROOT / "src",
            "/workspace/src",
            ignore=["**/__pycache__/**", "**/*.egg-info/**"],
        )
        .add_local_dir(ROOT / "tests", "/workspace/tests", ignore=["**/__pycache__/**"])
        .add_local_dir(ROOT / "tools", "/workspace/tools", ignore=["**/__pycache__/**"])
        .add_local_dir(
            ROOT / "benchmarks", "/workspace/benchmarks", ignore=["**/__pycache__/**"]
        )
        .add_local_dir(
            ROOT / "examples", "/workspace/examples", ignore=["**/__pycache__/**"]
        )
        .add_local_file(WORKLOAD, "/workspace/workload.json")
    )


image = with_workspace(base_image)

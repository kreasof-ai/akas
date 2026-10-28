"""LFM3 QAD Q4_0 on WebGPU/Vulkan against the same GGUF in llama.cpp Vulkan."""

from dataclasses import replace
from pathlib import Path
from tools.lfm3_benchmark import PROFILE as CUDA_PROFILE
from tools.model_benchmark import main

PROFILE = replace(
    CUDA_PROFILE,
    provider="webgpu",
    script=__file__,
    program=Path("build/lfm3-vulkan/program"),
    binary=Path("build/baselines/llama-b11946-vulkan/llama-server.exe"),
    optimized_options=dict(activation_quantization="mixed"),
    trials=1,
    chunk=512,
)

if __name__ == "__main__":
    main(PROFILE)

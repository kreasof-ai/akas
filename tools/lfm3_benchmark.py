"""Matched LFM3 QAD Q4_0 CUDA sweep against llama.cpp only."""

from pathlib import Path
from tools.model_benchmark import Profile, main

PROFILE = Profile(
    id="lfm3-1.7b",
    model="LiquidAI/LFM3-1.7B",
    source=dict(
        repo="LiquidAI/LFM3-1.7B-GGUF",
        revision="b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf",
        file="LFM3-1.7B-QAD-Q4_0.gguf",
    ),
    checkpoint_sha256="43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452",
    definition="akas.models.lfm3:LFM3",
    weight_format="q4_0",
    file_type=2,
    checkpoint=Path("build/checkpoints/lfm3-1.7b/q4_0.gguf"),
    program=Path("build/lfm3/program"),
    binary=Path("build/lfm3/llama.cpp/build-cuda-local/bin/llama-server"),
    vocab=128000,
    optimized_options=dict(
        optimized_attention=True,
        weight_arena=True,
        half_activations=True,
        q4_layout="planar",
        q8_ffn=True,
        q8_projections=True,
        fused_sandwich=True,
        fused_q8_ffn=True,
    ),
    trials=12,
    chunk=1024,
)

if __name__ == "__main__":
    main(PROFILE)

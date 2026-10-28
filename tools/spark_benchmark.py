"""Matched Spark Q4_K_M CUDA context sweep; see tools.model_benchmark."""

from pathlib import Path
from tools.model_benchmark import Profile, main

PROFILE = Profile(
    id="spark-x2.6-4b",
    model="XHToken/Spark-X2.6-4B",
    source=dict(
        repo="XHToken/Spark-X2.6-4B-GGUF",
        revision="83b47fda018fd14f51aa63341f78266f628af5a5",
        file="Spark-X2.6-4B-Q4_K_M.gguf",
    ),
    checkpoint_sha256="91cb2811aebb47ac916aa619d27d4cdd7f4143106b7917accb8ead4cd7f5cbd9",
    definition="akas.models.spark:Spark",
    weight_format="q4_k_m",
    file_type=15,
    checkpoint=Path("build/checkpoints/spark-x2.6-4b/gguf/Spark-X2.6-4B-Q4_K_M.gguf"),
    program=Path("build/spark/program-gguf"),
    binary=Path("build/baselines/llama.cpp/build-cuda/bin/llama-server"),
    vocab=131072,
    optimized_options=dict(
        rotary_cache=True,
        optimized_attention=True,
        embedding_location="host",
        weight_arena=True,
        fused_ffn=True,
        half_activations=True,
        compact_swa=True,
    ),
)

if __name__ == "__main__":
    main(PROFILE)

# Spark X2.6

`from akas.models.spark import Spark` reads the `spark2_5` architecture used by
Spark X2.5 and X2.6. The initial qualified checkpoint is XHToken/Spark-X2.6-4B.
Sources are pinned in `benchmarks/models/spark-x2.6-4b.json`.

The reader accepts BF16 safetensors, compressed-tensors FP8 safetensors with
per-output-channel weight scales, and GGUF. BF16 and FP8 safetensors are converted
one matrix at a time into FP16 resident weights. FP8 weight scales are applied
before conversion; execution uses floating activations, without dynamic FP8
activation quantization. This is FP8 checkpoint import, not native W8A8 execution.
Normalization and residual arithmetic use FP32, projections use FP16 operands
with FP32 accumulation, and KV is FP16. GGUF matrices retain their actual source
encodings, including mixed BF16, Q4_K, Q6_K, and Q8_0 in the official Q4_K_M file.
An explicit GGUF output head is used even when the original HF model ties it to
the embedding; quantization can give those two matrices different values.

The graph implements headwise sigmoid attention gates, exact GELU feed-forward
gating, grouped query attention, and per-layer rotary widths and frequencies.
Sliding attention uses a circular cache with a full prefill chunk of extra
capacity. All token positions and rotary phases remain absolute. Full attention
retains the complete requested context. The shared compiler, request mapping,
executor and device feedback own all execution.

CUDA with FP16 KV is currently supported. WebGPU, other KV precisions, INT4/INT8
safetensors and speculative decoding are not implemented. Tokenization supports
the checkpoint's ordered isolated splits, CJK segmentation and individual digits.
Single-user chat follows the checkpoint's default high-thinking template.
Multi-turn tool conversations and other thinking modes require explicitly
formatted prompts or token IDs.

```python
from akas.models.spark import Spark
from akas.compilation import compile_checkpoint

compile_checkpoint(
    Spark,
    "Spark-X2.6-4B-Q4_K_M.gguf",
    "build/spark/program",
    geometries=[(1, 1), (1, 32), (1, 256)],
    max_requests=1,
    max_context=131072,
)
```

`AKAS_SPARK_CUDA=1 python -m pytest tests/test_spark.py` qualifies independent
small-model equations, partial rotary, cache wrap, ragged chunks, request reorder
and slot reuse. `AKAS_SPARK_CHECKPOINT=/path/to/bf16` also compares tokenizer IDs
against the Hugging Face tokenizer implementation. These are distinct from
pretrained-model quality or full trajectory agreement with llama.cpp.

The full performance sweep only uses Q4_K_M: one request, prefixes 32, 512, 2048,
8192, 16384, 32768, 65536 and 130944, and 128 total outputs. It compares the same
checkpoint and synthetic input IDs against llama.cpp with FP16 KV. Prepare the
CUDA runtime libraries and `TENSOR_NVRTC_HOME` as described in the repository
setup, then run:

```sh
python -m tools.spark_benchmark compile --out build/spark/compile.json
python -m tools.spark_benchmark akas --out build/spark/akas.json
python -m tools.spark_benchmark llama --out build/spark/llama.json
python -m tools.spark_benchmark report --runs AKAS_RUN LLAMA_RUN --out build/spark/report
python -m tools.benchmark_archive publish
python -m tools.benchmark_archive check --objects --tracked
```

The default protocol uses one warmup and three retained samples per prefix.
Compilation, upload and phase loading are outside the timers. Native AKAS phase
rates and llama-server eval rates have different timing boundaries. Retained
output comparisons show cross-runtime trajectory agreement separately.

Follow `benchmarks/README.md` to publish artifact branches before the source
catalog, then prune verified local bundles. Never add run/report payloads or
compiled programs to the source branch.

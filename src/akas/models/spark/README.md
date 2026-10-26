# Spark X2.6

`from akas.models.spark import Spark` reads the `spark2_5` architecture used by
Spark X2.5 and X2.6. The initial qualified checkpoint is XHToken/Spark-X2.6-4B.
Sources are pinned in `benchmarks/models/spark-x2.6-4b.json`.

The reader accepts BF16 safetensors, compressed-tensors FP8 safetensors with
per-output-channel weight scales, and GGUF. BF16 safetensors use FP16 resident
weights. FP8 matrices retain their original E4M3 bytes and BF16 channel scales
on the GPU, and GGUF matrices retain their original packed encodings. Quantized
coefficients are decoded only in temporary compute tiles; no full FP16 weight
copy or persistent dequantized prefill cache is allocated. FP8 execution uses
floating activations, without dynamic FP8 activation quantization.
Normalization and residual arithmetic use FP32, projections use FP16 operands
with FP32 accumulation, and KV is FP16. GGUF matrices retain their actual source
encodings, including mixed BF16, Q4_K, Q6_K, and Q8_0 in the official Q4_K_M file.
An explicit GGUF output head is used even when the original HF model ties it to
the embedding; quantization can give those two matrices different values.

The graph implements headwise sigmoid attention gates, exact GELU feed-forward
gating, grouped query attention, and per-layer rotary widths and frequencies.
Sliding attention uses a circular cache. Compact mode retains one window and
stages the current K/V chunk in a workspace reused across layers. Other programs
reserve a full prefill chunk of extra cache capacity. All token positions and rotary phases remain absolute. Full attention
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

For the faster L4 configuration, compile and run with a larger chunk:

```sh
python -m tools.spark_benchmark compile --optimized --chunk 2048 --program build/spark/fast --out build/spark/compile-fast.json
python -m tools.spark_benchmark akas --chunk 2048 --program build/spark/fast --out build/spark/akas-fast.json
```

The equivalent model options are `rotary_cache=True`,
`optimized_attention=True`, `embedding_location="host"`, `weight_arena=True`,
`fused_ffn=True`, `half_activations=True` and `compact_swa=True`, with
`prefill_chunk=2048`.
Quantized weights retain their original bytes in one aligned GPU allocation;
another allocation holds persistent state without per-layer allocation padding.
The separate input embedding uses pinned CPU storage, matching llama.cpp's
placement; the output head and all transformer weights remain on the GPU.
Host embedding placement requires an untied GGUF head.

Compact FP32 sine/cosine tables reconstruct absolute rotary phases from fine
and coarse positions. Packed matmul decodes temporary tiles in shared memory;
fused feed-forward projections avoid global gate/up intermediates. Prefill
activations are stored at the following tensor-core operation's FP16 precision,
with FP32 reductions and residuals. Residual addition and normalization share a
kernel, and QKV splitting, rotary rotation and cache writes share another.
No persistent FP16 weight expansion is used.
Optimized attention shares K/V loads across grouped decode heads and prefill
tiles. CUDA phases share scratch buffers and identical loaded kernels on the
device's ordered stream. Phase outputs and persistent state remain distinct.
These model options are disabled by default. Compilation and uploads remain
outside benchmark timers.

The former `prefill_weight_cache` option has been removed. Recompile older
programs that enabled it, and FP8 programs built for the former FP16 import
layout. Historical benchmark records remain archived.

Run the expanded independent checks with
`AKAS_SPARK_CUDA=1 python -m pytest tests/test_spark.py tests/test_spark_performance.py`.
The high-context operator checks include inactive requests, circular cache wrap,
nonfinite padding, odd projection dimensions and partial rotary positions.
The mixed GGUF model reference also checks fused prefill, mapped input embeddings,
byte-preserving weight slices, request reorder and slot reuse. Shared resource
ownership tests are in `tests/test_phase_memory.py`.

Estimate useful prefill MFU from a generated report with:

```sh
python -m tools.spark_mfu PATH/TO/report.json --checkpoint PATH/TO/MODEL.gguf --peak-tflops 121 --out build/spark/mfu.json
```

The default 121 TFLOP/s is NVIDIA L4's dense FP16 peak, without sparse doubling
([NVIDIA Ada architecture whitepaper, Appendix D](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)).
The estimate counts dense projections, one final-row vocabulary projection,
and useful causal QK/PV pairs, including the bounded sliding windows. It excludes
embedding lookups, padding, dequantization, elementwise operations and cache/control
traffic. It counts all model layers even where intermediate prefill chunks prune
final-layer outputs. This is a useful-work estimate, not measured instruction
utilization; matmul-only microbenchmark throughput is not whole-model MFU.

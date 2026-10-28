# LFM3

`LFM3` loads the official LiquidAI/LFM3-1.7B BF16 safetensors,
LFM3-1.7B-FP8 safetensors, or LFM3 GGUF through one declaration:

```python
from akas.models.lfm3 import LFM3
from akas.compilation import compile_checkpoint

compile_checkpoint(
    LFM3,
    "build/checkpoints/lfm3-1.7b/q4_0.gguf",
    "build/lfm3/program",
    checkpoint_options={"tokenizer": False},
    geometries=[(1, 1), (1, 32), (1, 512)],
    max_requests=1,
    max_context=131072,
    model_options={"prefill_chunk": 512},
    provider="cuda",
    target="sm_89",
)
```

CUDA execution uses FP16 KV, FP32 residuals and normalization, and FP16
projection operands with FP32 accumulation. BF16 source matrices are imported
into FP16 execution. FP8 matrices remain E4M3 bytes with their original BF16
128×128 block scales; consuming projections decode tiles locally. Dynamic W8A8
activation quantization is not enabled. GGUF matrices retain their original
individual encodings, including the QAD Q4_0 file's Q6_K tied embedding/head.
There is no persistent floating copy of quantized matrices.
The optimized GGUF path separates Q4 payloads from their original FP16 scales
without changing coefficients or the packed byte count. It retains one packed
copy. FP16 activation buffers keep the projection's existing operand precision;
residuals and normalization arithmetic remain FP32.
The benchmark also enables symmetric Q8 activation blocks of 32 for large Q4_0
prefill projections (`q8_ffn=True, q8_projections=True`). Integer tensor cores
produce exact integer partial sums, followed by the original weight scales,
activation scales and FP32 accumulation. Decode and short prefill use floating
activations. This optional execution choice is separate from FP8 weight loading.
`fused_sandwich=True` combines a sandwich norm, residual addition and the next
norm in one kernel, retaining FP32 residuals and the two normalization equations.
`fused_q8_ffn=True` also packs the FFN activation in the projection kernel,
avoiding a separate activation buffer write and packing pass. Both fusions are
enabled by the benchmark's optimized profile.

The architecture combines seven-tap gated short convolutions, Q/K RMSNorm,
headwise sigmoid attention gates, SwiGLU, and optional sandwich norms. Sliding
attention retains a bounded ring plus one prefill chunk; full attention retains
the requested context. Partial RoPE rotates pairs across the full head halves,
using the full head dimension in its frequency exponent. The GGUF frequency
factors and safetensors rotary configuration map to the same operation.
Compact fine/coarse rotary tables preserve long-position phases. Request reuse
resets positions and convolution histories; attention masks hide stale KV.

The tokenizer supports literal user text and the official single-user chat
format with the template's default thinking mode. Tool calls, packed documents,
media inputs and additional conversation-template options are outside this text
adapter. The published model context limit is 262,144 tokens.

## Single-user comparison

The pinned [model sources](../../../../benchmarks/models/lfm3-1.7b.json) include
all three formats. The [campaign](../../../../benchmarks/campaigns/lfm3-1.7b-single-user-v1.json)
benchmarks only QAD Q4_0 against llama.cpp CUDA, on identical checkpoint bytes
and deterministic prompt IDs. Prefix lengths are 32, 512, 2,048, 8,192, 16,384,
32,768, 65,536 and 130,944. Each request starts empty and generates 128 tokens:
prefill predicts token one, and decode commits the remaining 127 tokens.

```sh
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
python -m tools.lfm3_benchmark compile --chunk 1024 --optimized --trials 12 --out build/lfm3/compile.json
python -m tools.lfm3_benchmark akas --chunk 1024 --out build/lfm3/akas.json
python -m tools.lfm3_benchmark llama --out build/lfm3/llama.json
python -m tools.lfm3_benchmark report --runs AKAS_RUN LLAMA_RUN --out build/lfm3/report
```

Use a llama.cpp build with LFM3 support; the retained run pins its source and
binary digests. Defaults retain three repetitions after one warmup per shape.
Both phase rates and output agreement are reported. AKAS synchronized native
phases and llama-server eval phases have different timing boundaries; phase
rate comparisons do not establish identical kernel or complete-request timings.
Output capture restarts the request to restore all recurrent histories.

The retained [NVIDIA L4 comparison](https://github.com/kreasof-ai/akas/blob/2f19a43426682bc9dbeff464b988b2544cec000c/benchmarks/reports/20261027T185932Z-b917d23ac845/README.md)
passes [both phase checks at all eight prefixes](https://github.com/kreasof-ai/akas/blob/2f19a43426682bc9dbeff464b988b2544cec000c/benchmarks/reports/20261027T185932Z-b917d23ac845/PARITY.json):
prefill medians improve 1.9–28.9% and decode medians improve 7.6–10.7% over the
pinned llama.cpp Q4_0 baseline. See the [throughput chart](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028073053545412-431db636/analysis/throughput.svg)
and [validation, precision and reproduction notes](https://github.com/kreasof-ai/akas/blob/c1d260dc031cc412ab59180286c5dda978f6cee3/benchmarks/runs/20261028073053545412-431db636/analysis/VALIDATION.md).
These are phase-rate results; generated sequences differ at some prefixes, as
recorded in the report's `output_comparisons`.

Independent model qualification covers BF16, FP8 and GGUF, causal convolution,
full-head partial rotary pairs, sliding-ring wrap, ragged lengths, reordered
requests and reuse. Run it with:

```sh
AKAS_LFM3_CUDA=1 python -m pytest tests/test_lfm3.py
```

`python -m tools.lfm3_qualify` seals this qualification and the CUDA Q8 packing
tests as a benchmark record. Q8 model checks require greedy-token agreement and
bound relative vector and normalized peak logit error to 3% against independent
quantized equations. Projection primitives have their own tighter numerical
checks; packing tests check exact bytes, scales and sums, including zero blocks
and rounding ties. Full pretrained output agreement is recorded separately.
The fused FFN primitive compares encoded values within one quantization level
of an independent FP64 projection reference, since FP32 accumulation can straddle
a rounding boundary; its scale tolerance is 0.2%.

# LFM2 qualification

The current composed floating runtime and pretrained CUDA/hardware-WebGPU
measurements are recorded in [performance repair](performance-repair/README.md).
The results below describe the earlier serial reference path and its diagnostic
fixture; they remain useful as correctness evidence.

This records the readable reference path, not a throughput claim for the optimized
packed implementation. Model composition stays in the three LFM2 model files;
loading, state management, operators, codecs, and timing are shared components.

CUDA (L40S) and WebGPU (Mesa llvmpipe CPU Vulkan adapter) executed batches
1/4/16/64/256 with FP16 and FP8 KV. Every case exercised both GGUF and equivalent
safetensors, mixed short prompts, 128-token prompts, masked slots and reset.
Dedicated tests compare FP8 payload bytes/scales against Torch E4M3, including
rounding ties, and verify partial reset preserves sibling generation state.
Existing 143 packed arithmetic hashes and 309 semantic requirements remain unchanged.

Pretrained LiquidAI/LFM2-350M revision
`f37d3f5c8c5484bc01dad379a595cf4c68c4e70e` was checked at batch 1.
F32 CUDA logits matched Transformers eager F32 at relative RMS error 4.39e-7.
FP8 CUDA and final portable WebGPU matched independent NumPy equations at relative
RMS errors about 1.80e-5, with identical greedy tokens. The installed wheel also
loaded and generated on Tensor's pinned WebGPU 0.29.0 with compiler, Torch and
Transformers imports blocked. Downloaded pretrained weights are not committed.

FP8 long-context tests require relative vector error below 0.002, maximum absolute
error below 0.003 and identical greedy tokens. Small F32 accumulation differences
can cross BF16/FP8 rounding boundaries; per-logit relative error near zero is not a
useful quantized-vector criterion. The codec's byte-level oracle remains exact.
FP16 logits retain per-element numerical checks.

The recorded performance matrix uses the **synthetic two-layer diagnostic fixture**,
FP8 KV, CUDA graphs, prompt lengths 32/2048 and all five batch sizes. Each case has
an independent numerical preflight, one excluded warmup and three repetitions.
Treat it as an illustrative harness report, not a pretrained performance target:
other qualification jobs ran on this machine during measurement. Re-run with the
same device/checkpoint/protocol before comparing tuning candidates.

WebGPU FP16 cache values are rounded to FP16 but backed by portable F32 buffers;
FP8 uses four packed bytes per word plus scales. No physical FP16 memory-saving
claim is made. For graphics GPU performance, see the composed floating measurements linked above. Context and batch
allocations must fit adapter binding/buffer limits. See
[the example](../../../examples/lfm2/README.md) and
[the daily tuning harness](../../guides/performance-harness.md).

# LFM3-1.7B implementation and measurement evidence

The single-user workload starts each request empty, uses deterministic identical
prompt IDs at prefixes 32, 512, 2,048, 8,192, 16,384, 32,768, 65,536 and 130,944,
and generates 128 tokens. Prefill predicts the first token; decode commits 127.
Only the official QAD Q4_0 GGUF is benchmarked. Checkpoint, llama.cpp source and
binary digests are recorded in the associated runs. The measured GPU is NVIDIA
L4; the results do not establish performance on a different GPU.

## Storage and execution

Quantized matrices have one packed device copy. The Q4 payload/scale separation
is an exact repack with unchanged byte count; Q6_K embedding/head storage remains
in its original layout. Projection tiles decode coefficients locally. There is
no persistent floating weight cache. FP16 KV and FP32 residual arithmetic remain
in use. Large Q4 prefill projections use block-32 Q8 activations and integer
partial products followed by FP32 scaling. The fused FFN packs its activation
before leaving the projection kernel. Decode and short prefill retain floating
activations. BF16 source matrices import into FP16 execution; FP8 source matrices
remain E4M3 with BF16 block scales and use W8A16 projections.

## Numerical checks

- `20261027221231165214-6392dfc9`: passed independent BF16, FP8, GGUF,
  aligned FP16-activation GGUF, Q8 FFN and Q8 projection model checks, plus CUDA
  Q8 packing. Tests cover prefill/decode, ragged and reordered requests, sliding
  cache wrap, convolution history and slot reuse, and official tokenizer/template
  agreement. Independent NumPy equations supply the reference.
- `20261028071546944954-9fb5e2d3`: passed the final fused Q8 FFN model path and
  both CUDA Q8 codec precisions, with the preceding all-format qualification
  explicitly recorded as its base.
- `20261027221834452796-2ba99389`: passed bitwise comparisons with the original
  convolution and packed projection kernels. Includes decode, chunks 32/512,
  inactive/ragged state, paired FFN, FP32 inputs and already-rounded FP16 inputs
  with a partial final tile. Original source and comparison script are retained.

Q8 model checks require equal greedy tokens, relative logit-vector error below
3%, and peak logit error below 3% of the reference peak. Independent attention
rounding can cross a Q8 boundary. Projection primitives have tighter local
references. The fused producer allows one quantization level against FP64
projection equations and 0.2% relative scale error. Standalone activation packing
checks exact bytes, scales and sums, including zero blocks and rounding ties.
Full pretrained generated-token agreement is reported separately from timing.

The affected host checks passed: 59 passed, 11 skipped and 3 deselected. The broad
host suite completed with 933 passed, 1,017 skipped and 10 failures. All ten
failures were also reproduced on the unchanged checkout: two existing frozen
WebGPU/packing computation guards, four stale convolution commit-name assertions,
and four executor test doubles missing the phase/read keyword arguments. Logs
include the baseline reproductions; these failures were not hidden or treated
as passes. CUDA checks are opt-in and are recorded above separately.

## Compilation and timing

`seed-profile-final.json` records the qualified decode/32-token schedules seeded
from the earlier 12-trial search. Final compilation used a one-trial allowance
for the remaining operations, two timing samples and three repeats. This is not
a claim that the whole investigation required only one trial. `reproduce.py`
replays that final recipe with the retained seed profile.

Both measured runtimes use one warmup and three timed repetitions per prefix.
The acceptance check compares medians for both phases at every prefix. AKAS
synchronizes native token phases; llama-server reports eval phases. Their phase
boundaries differ, so the table is a phase-rate comparison rather than identical
kernel or complete-request timing. Output capture restarts the entire request,
including convolution histories, outside timing. All model phases load before
measurement. Reported allocation includes the program's weight, state and
scratch buffers; there is no additional dequantized weight copy.

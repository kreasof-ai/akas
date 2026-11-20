# Convolution

`conv_verify` produces outputs and a journal of every history prefix without
changing persistent history. `conv_commit_prefix` restores the history after
the accepted input count; zero acceptance leaves it unchanged. Both use shared
CUDA/WebGPU bodies. This prefix-journal contract differs from GDN's existing
`conv_commit`, which replays raw projected inputs into its history.

## Purpose

LFM-style gated short convolution with persistent history. Conceptually: `convolution(x,
weights, history, enabled) -> (history, out)`. Its gating/history contract differs from
GDN preparation convolution.

## Backend choice

**Recommendation: Keep the current shared computations.**

- **Reason to share:** Both providers already select `convolution` or `convolution_flat`
  from `kernel.py`. They use parallel channel work and serial token/history updates with
  no explicit backend intrinsic. Flat decode and batched chunks are layout/state
  contracts.
- **Reason to split:** Separate backend bodies would duplicate the history update and
  gate arithmetic. Specialize only if a new measured convolution algorithm warrants it,
  while retaining the same history order and inactive-token behavior.

The six-file structure is implemented. Consolidation preserves the existing
registered support; shared bindings alone do not establish new device support. See the [shared authoring and backend
guide](../../../../docs/operations.md)
for the fixed file roles and the distinction between construction,
compilation, execution and performance.

## Current code

| File | Responsibility |
| --- | --- |
| [__init__.py](__init__.py) | Package boundary. |
| [definition.py](definition.py) | Public composition, semantic/output contracts, execution policies, registrations and eligibility. |
| [kernel.py](kernel.py) | Shared TileLang computations and inline helpers. |
| [cuda.py](cuda.py) | CUDA specializations, shared bindings and approved CUDA registration inventory. |
| [webgpu.py](webgpu.py) | WebGPU specializations, shared bindings and approved WebGPU registration inventory. |
| [validation.py](validation.py) | Independent host expectations for qualification. |

Both reference/chunk and flat-history bodies are in `kernel.py`. The empty legacy CUDA package has been removed.

## How to extend

Start with the shared body and state contract. A different tap count or gate layout
needs explicit shapes and independent expectations. Do not treat GDN's convolution as an
interchangeable variant merely because both use history buffers.

Keep the typed tensor inputs → explicit results format and lazy compiler imports.
Declare new choices and effects in `definition.py`; it is the discovered
registration boundary. Keep numerical expectations in `validation.py`.

## Validation

Check chunked execution against repeated token execution, history ordering, short/empty
enabled prefixes and channel tails. See [state prefill
tests](../../../../tests/test_state_prefill.py) and [LFM reference
tests](../../../../tests/test_lfm2_reference.py).

## ZONOS3 and ZAC

The causal depthwise FIR uses a zero-padded input prefix and one bias per channel for full-utterance decoding, while `causal_depthwise_state` carries `(taps - 1)` per-channel history across multi-slot prefix-stream chunks. Streaming multi-resolution ZAC synthesis decomposes into `istft_polar` (log-magnitude/phase to real/imaginary spectrum), Fourier-Hann basis projections, `istft_ola` and `istft_mix_commit` (two-branch overlap-add and 127-tap FIR mixer with persistent overlap/mixer state), and `sinc_resample_limit` + `sinc_resample_commit` (44,160 -> 44,100 Hz sinc resampling, headroom gain, soft-knee limiter, and 322-sample resampler tail commit). CUDA and WebGPU share the portable computations.

## Strided 2D convolution

`image_patches` extracts zero-padded square patches from contiguous
`(width, height, channels)` floats. It writes one row per output position in
`(channel, kernel height, kernel width)` order for the existing tiled linear
operation. CUDA and WebGPU share the gather. The audio frontend uses stride
two and 3×3 filters.

WebGPU `single_channel_conv2d` shares a short input patch across filters, avoiding
a separate gather for the first audio convolution. `image_patches_half` stages
the wider convolutions in tap-major channel groups to make the gathers
contiguous and rounds to F16 before the tiled projection. Each gather uses a
NumPy sliding-window reference.

`implicit_half_conv2d` uses the same tap-major weights. It gathers the
zero-padded F16 input directly into the shared tile of the half-precision
matrix multiply. A tile covers channels from one spatial tap. The independent
sliding-window reference exercises stride, padding and filter ordering using
operands whose short F16 accumulations are exactly representable.

Return to the [operation index](../README.md).

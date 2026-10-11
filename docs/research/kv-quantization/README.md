# KV codec extraction audit

Baseline: `b45791c48629023fd7a6070c2c3f247000be0657`.

`files.csv` maps every changed source and each new file. The original
`attention/cuda/kv_fp8.py` is split into `attention/cuda/quantized.py` for
preparation and attention, and `kv_quantization/cuda/fp8.py` for conversion.
`units.csv` records original definitions and the extracted writer/decoder targets.
`coverage.json` verifies all targets, frozen specializations and unchanged
arithmetic hashes. Original migration inventories remain historical snapshots;
this map supplies the successor for their former `kv_fp8.py` target.

Storage definitions moved from shared state allocation into the named codec's
compiler-free layout. The existing FP8 payload, scale shapes, ordering and dtypes
remain identical. Main block-FP8 requirements, allocation, composition and selection
resolve codec names through the shared format interface. Split, workspace, indirect
ownership and Hopper attention reuse the extracted FP8 conversion macros while
retaining their qualified layouts. Numerical source export uses the shared operation
dependency closure so codec edits invalidate producer caches.

The original 143 arithmetic hashes are unchanged. `tests/kv_extraction.py` expands
actual codec macros into their former callers, substitutes the original FP8 layout,
and normalizes integer address expressions to compare with those frozen hashes.
It removes selection/layout plumbing, but does not replace encoding, decoding,
normalization, rotary or attention math with new golden values.

Qualification uses synthetic inputs and independent Torch FP8 references. Existing
cache write/decode cases pass before and after extraction; causal prefill and
speculative split attention are checked as well. Standalone codecs exercise channel
counts 128/256/512, KV head counts one/three, positions, inactive rows, exact stored
bits/scales and BF16 decode results. CPU tests check another format's packed layout
and automatic discovery/selection, unknown formats, missing providers and rejection
of unavailable packed loads. The test-only INT4 descriptor has no INT4 arithmetic
implementation and makes no numerical support claim.

No pretrained weights were downloaded or full-model throughput measured. Hopper
hardware execution remains unqualified locally; its extracted numerical definitions
retain the frozen arithmetic checks. Current attention geometry and the one-metadata
pair ABI remain explicit. More complex layouts require compatible bindings; format
availability does not override those constraints. Read the API in
`src/akas/ops/kv_quantization/README.md`.

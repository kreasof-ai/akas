# Merged executor migration

`files.csv` maps every tracked execution/profile file from the baseline revision
in `baseline.json` to its current destination. The old profile package and the
three execution-mode packages are removed; empty initializers merge into their
new package owners. Compiler profiles become recipes named for numerical layouts.

The shared executor owns admission, release, validation, prefill/decode dispatch,
thread/session checks and teardown. Bindings preserve numerical forward plans,
state layouts, direct-forward APIs and speculation commit semantics. Required
kernels move with their bindings; schedules move with operations; compiler routing
moves into the compilation package. The original kernel arithmetic fixture remains
the verification baseline.

Existing implementation-bound bundles require rebuilding. CLI profile keys and
bundle schemas remain compatible. Dynamic shapes and adaptive variant selection
are separate work; this migration preserves existing compiled geometries.

Validation: the CPU suite passed 437 tests with 397 hardware checks skipped;
three additional scheduler cases passed for the merged bindings. All 28 LFM2
CUDA request checks and 35 Qwen CUDA kernel/speculation checks passed on L40S.
The installed compiler-free wheel generated three interleaved synthetic requests.
The 143 frozen arithmetic units match their original AST baseline.

A supplementary tiny synthetic BF16 whole-model build failed in TileLang layout
inference on the pre-merge revision, before forward execution. It does not provide
whole-model BF16 parity qualification. No pretrained checkpoints were downloaded.

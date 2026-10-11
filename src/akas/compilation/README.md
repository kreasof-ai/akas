# Shared compilation

Compilation belongs to the installed AKAS package. Compiler dependencies remain
optional and are imported only while rendering or compiling kernels. Inference
can import the package and consume bundles without the compiler extra.

`plan.py` describes `Source`, `Kernel` and `Bundle`. `build.py` compiles these
same declarations for every model, checks source/provider/target/compiler identity,
records artifact checksums and publishes a completed manifest atomically.
`bundles.py` copies checked artifacts when deriving a new execution bundle.

Profiles describe execution contracts, not model families:

| Profile | Inputs and responsibility |
| --- | --- |
| `packed` | Loaded packed checkpoint; CUDA/WebGPU rows and numerical profile |
| `batched` | Loaded checkpoint geometry; BF16 hybrid batches, verification or draft |
| `batched-bank` | Several batched geometries plus optional speculation controls |
| `hybrid.decode` | Loaded checkpoint geometry matching the fixed-layout block-FP8 kernels |
| `hybrid.prefill` | The same FP8 contract with chunked prefill and cache precision |
| `hybrid.draft`, `hybrid.draft-prefill` | Loaded draft checkpoint and a checksummed target bundle |

`packed`, `batched`, `hybrid.decode` and `hybrid.prefill` expose `plan()` for
inspecting kernel declarations before building. Model definitions supply checkpoint readers;
profiles receive the resulting objects and never import model-family definitions.
Adding a model that matches an execution contract requires configuration, weight
roles and decoder composition, not another producer implementation.

```sh
akas compile --definition akas.models.lfm2.model:LFM2 \
  --checkpoint model.gguf --profile packed --out build/model \
  --config '{"context":384,"target":"sm_89"}'

akas compile --definition akas.models.qwen35.model:DenseEngine \
  --checkpoint /path/to/checkpoint --profile batched --out build/model \
  --config '{"slots":8,"chunk":32,"context":4096,"target":"sm_89"}'
```

Python callers can pass a loaded checkpoint to `profiles.packed.plan()` or
`profiles.batched.plan()`, inspect the result, then call `build.build()`.
`compile_checkpoint()` loads through a supplied definition and invokes the selected
profile. Checkpoint reader options, such as a draft branch, are supplied separately.

`transforms/` selects reusable attention, projection, expert, recurrent and device
specializations over existing bundles. Kernel source factories remain in
`akas.kernels`; candidate measurements and searches remain in `benchmarks`.

Fixed-layout FP8 kernels currently support one explicit geometry, checked in
`profiles/hybrid/geometry.py`. A different width or state layout needs a compatible
kernel/profile implementation; compilation never substitutes another model's
geometry silently. Measured device specializations retain their qualification
limits. Existing schema names retain compatibility with runtime bundle readers.

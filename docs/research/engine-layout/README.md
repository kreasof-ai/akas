# Engine ownership sweep

This ownership migration is implemented. The baseline is
`a6ed065`. It covers every non-generated file in `src/akas/{ops,memory,kernels,
generation,execution,artifacts}`: 97 files and 15,686 Python lines, plus explicitly mapped companion profile and packaging files. The existing
numerical contracts and manifest schemas remain the preservation baseline.

## What is wrong with the current layout

- `kernels/hybrid/hybrid.py` implements 19 different opcode branches in one
  815-line factory: projections, embeddings, normalization, expert routing,
  attention, token selection, and GDN. `hybrid_batch.py` and `hybrid_prefill.py`
  repeat this grouping for other geometries.
- `kernels/packed/{ggml,cuda,webgpu}.py` mix storage decoders, projections,
  attention, normalization, embedding, row control, and reductions. The WebGPU
  file is 1,415 lines. Provider and weight format are currently acting as owners
  for unrelated mathematical operations.
- `ops/` groups host composition by operation, while the matching arithmetic is
  scattered across `kernels/`. `ops/ffn.py` also mixes dense FFN and routed experts.
  Its `hybrid_state.py` allocates memory rather than composing an operation.
- `memory/recurrent_journal.py` loads bundles, creates kernel plans, and captures
  graphs. Those actions belong to execution. `memory/estimates.py` supports Qwen
  geometry only; it does not share a layout definition with the allocator.
- `generation/greedy.py:25–40` submits WebGPU/CUDA work and advances persistent
  execution state. Both MTP classes own executor coordination and commit/repair;
  `batched_mtp.py` also owns admission, graph rounds, and executor destruction.
- `artifacts/{packed,hybrid}.py` mix required specializations with implementation
  fingerprints. Fingerprint functions are also scattered across execution files.
  The packed executor imports profile constants from a CUDA kernel module.
- `execution/{packed,hybrid,batched}` describe different things: a weight format,
  a decoder architecture, and a batching strategy. Their actual resource
  contracts are independent handles, fixed resident slots, and compact execution
  batches. Giving these the same base class would hide important state semantics.

There are also fixed Qwen dimensions inside supposedly general factories and
requirements: width 2,048, 256 experts, top-8, and specific GDN dimensions. Moving
those bodies does not make them general. Keep their supported geometry explicit;
expand it separately with numerical validation.

## Implementation owners

```text
src/akas/
  models/blocks.py                 shared decoder declarations
  ops/
    attention/                    composition and attention kernels
    linear/                       composition and projection kernels
    moe/                          routed composition and expert kernels
    ffn/                          dense/packed FFN composition and activation
    gdn/                          composition, scan, journal and restore kernels
    convolution/                  short-convolution composition and kernels
    normalization/                composition and normalization kernels
    quantization/                 storage decoders, operand conversion, scratch validity
    reductions/                   shared ordered/tree/subgroup reduction helpers
    vocabulary/                   embedding, output projection and token reductions
    sequence/                     row mapping, lengths, last rows and advancement
    residual/                     residual addition
    mtp/                          MTP join/cast kernels
    speculation/                  device control kernels for a speculative round
  memory/
    state.py                      BufferSpec and allocation/cleanup
    layouts.py                    persistent-state geometry
    estimates.py                  storage estimates from that geometry
  execution/
    plan.py                       ordered calls and argument binding
    independent/                  resident.py, request.py, plan.py, workspace.py, greedy.py
    fixed_slots/                  resident.py, decode.py, prefill.py, workspace.py,
                                  draft.py, verify.py, repair.py, pools.py, speculative.py
      optimizations/              checked replacements for this execution contract
    compact_batch/                resident.py, batch.py, plan.py, workspace.py,
                                  draft.py, speculative.py, speculative_graph.py, journal.py
  generation/                     prompt/termination policy, acceptance, output lookup
  profiles/
    independent/                  requirements, parameters, selection, factory routes
    fixed_slots/                  requirements and factory routes
    compact_batch/                 factory routes
  artifacts/
    kernels.py                    logical IDs, integrity, loading and partial cleanup
    fingerprints.py               shared identities for contracts and extensions
  compilation/source.py           one source-export helper for selected factories
```

Each operation has `compose.py` when it composes several calls, plus its actual
kernel implementations. CUDA variants live under `cuda/`; WebGPU implementations
exist only where supported. Precision, layout, and hardware variants have named
files. Do not create empty symmetric provider trees or one file per small helper.
A shared fused projection stays one implementation even when both linear and FFN
composition use it.

The `profiles/*/factories.py` files are routing tables and validation. They must
contain no TileLang arithmetic. Every opcode branch in the old mixed factories
has an operation target in the unit map. The source-export wrappers merge into
one compilation helper; runtime imports must not execute compiler imports.
Profile requirements must be accessible to producer and consumer without loading
TileLang. Measured parameter tables remain explicit profile data.

Public `akas` model names remain unchanged. Internal executor names and imports
can change; legacy manifest schema strings and historical source paths retain
meaning and must not be rewritten as if they were current import paths.

## Complete file and code routing

- [File map](files.csv): every scoped source and companion file, SHA-256, explicit
  destinations, and action. Splits and merges are allowed; no file is unassigned.
- [Unit map](units.csv): definitions, state, opcode branches, dispatcher glue,
  source-export merges, and generation/device blocks, with original line ranges.
  Fingerprint definitions merge into one shared closure per execution contract.
- [Reference checklist](references.csv): imports and embedded module references
  from live source, benchmarks, examples, tools, and tests. This includes
  factory/provenance strings, not just import statements.
- [Coverage](coverage.json): baseline, counts, all targets with multiple inputs,
  and the interpretation of overlapping dispatcher/branch ranges.

Imports, module descriptions, and comments move with their implementation.
Class methods stay with their class. A dispatcher row owns selection, validation,
shared prerequisites, and fallbacks; its opcode rows own the arithmetic within
its enclosing range. Shared prerequisites must be resolved explicitly before
extracting any branch, rather than copying a whole old factory into each target.

`generation/greedy.py` retains its public prompt/result behavior. Its provider
branch moves to the independent executor, including position advancement. Keep
EOS, output-limit, context, and final-position behavior exactly as recorded.
MTP orchestration moves intact first: fixed-slot and compact-batch acceptance,
rollback, and repair have distinct execution contracts. Pure acceptance policy
and output lookup remain in generation.

`memory/estimates.py` now derives storage from the allocator
specifications, including FP8 KV scales and format-dependent byte sizes. Its
existing calling contract and rounding behavior must be preserved.

## Caller changes beyond these six directories

The reference checklist includes callers outside this inventory. Model entry
points, compilation profiles/transforms, scheduling adapters, benchmark factory
strings, optional installed-consumer checks, and tests must follow the new owners.
No model receives copied engine or kernel bodies.

Producer profile files should use the same execution-contract names:
`compilation/profiles/packed.py` becomes `independent.py`, `batched.py` becomes
`compact_batch.py`, and `hybrid/` becomes `fixed_slots/`. Pure requirement and
factory descriptions go in shared `profiles/`; compilation remains responsible
for producing bundles. These companion routes are included in the file map. Move JSON resources with
their owning profile, updating package-data rules and source-relative lookup paths.

## Preservation checks

1. Preserve source hashes and reconcile every row against a dirty-checkout backup.
2. Compare moved definitions and opcode arithmetic against the baseline AST,
   allowing reviewed import/name/ownership changes. Preserve generated kernel
   code, numerical parameter choices, rounding, reduction order, and layouts.
3. Update factory names and dependency modules in compilation fixtures explicitly.
   Keep opcode, parameters, specializations and selected variants frozen; do not
   regenerate fixtures from the implementation being tested.
4. Verify request/slot ownership, position advancement, stale scratch invalidation,
   prefix commit/repair, partial-load cleanup, and graph resource lifetimes.
5. Rebuild fingerprint-bound bundles; producer and independent wheel consumer must
   agree, with compiler imports blocked. Historical evidence hashes stay frozen.
6. Run the CPU suite and CUDA request/training-free inference checks. Run the
   producer/consumer package workflow on GitHub. Qualification of Qwen, Hopper,
   and physical WebGPU remains limited to the hardware evidence actually tested.
7. Check every target exists, every split unit has an owner, every merged input
   was incorporated, and old paths contain no leftover implementation aliases.

## Completed checks and scope

All 111 source/companion files have existing targets, verified baseline SHA-256
values, and no leftover implementation files at moved paths. The final map has
304 ownership units. The 143 kernel functions and opcode bodies match the
baseline AST, allowing import-path changes. Frozen compilation fixtures retain
kernel keys, specializations and schedule arguments; source locations and the
shared dependency closure are updated explicitly.

The CPU suite passed 410 tests with 397 hardware/reference skips. On the local
L40S, the request suite passed with both default/optimized CUDA profiles and graph
replay enabled/disabled; all 17 Qwen kernel numerical tests passed. The wheel and source distribution were rebuilt. An isolated consumer matched
producer fingerprints, loaded the CUDA bundle, ran forward and graph generation
with compilation imports blocked, and contained the relocated profile data.
No full pretrained checkpoint or physical Hopper/WebGPU qualification is claimed.

This migration preserves shape-specialized execution. Dynamic kernel variants,
beam search across workload ranges, runtime selection and a possible native
executor remain the next performance implementation, described in the
[engine architecture](../../architecture.md#compiled-runtime-direction).

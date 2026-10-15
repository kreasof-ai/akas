# Execution

`Executor.from_program()` opens a checksummed compiled program with one request-slot
pool. Model declarations use this same constructor through `models/definition.py`.
Runtime execution does not import kernel compilers or reference frameworks.

| File | Responsibility |
| --- | --- |
| `executor.py` | Admission, validation, phase dispatch, verification and accepted-prefix commit |
| `batch.py` | Semantic request inputs and nearest qualified phase geometry |
| `forward.py` | Load kernels, bind graph calls and capture/replay forward execution |
| `plan.py` | Bind ordered calls to Tensor executables |
| `workspace.py` | Allocate scratch/state, check view lifetimes and release owned resources |
| `state.py` | Slot lifecycle and deferred recurrent journals |
| `replay.py` | Capture proposal, verification, acceptance, commit and repair as one round |

Slots identify persistent state; execution rows can be reordered, padded or reused.
Weights/state are shared across phase geometries. Scratch is reused only when graph
lifetimes do not overlap. Borrowed views release their own descriptors, never base
storage. Admission and release reset only the state declared by the model graph.

`decode()` schedules one step. `decode_steps()` replays a fixed request group on the
device with a single control upload and optional final readback, for throughput
measurement. Both consume the same compiled decode phase.

`generation/speculative/decoding.py` owns the proposer-independent transaction flow.
`models/qwen35/mtp.py` supplies one head implementation. Captured rounds and deferred
state use the common executor rather than a separate model runtime.

All model families open the same executor. Legacy numerical bindings, independent
request adapters and token-step runtimes have been removed. Prefill uses qualified
chunk sizes, skips output-only work on intermediate chunks, and computes final
outputs through the normal phase. Speculative rounds select qualified windows
and cache captures by both row count and window width.

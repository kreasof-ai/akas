# Forward-plan migration audit

Baseline: `5bd9da2ca9af1514b51fbb246ecc5ae19e612b7e`.

`files.csv` maps every changed tracked file and every newly added file to its
current target. Operation routing tables were split among operation-owned
`implementations.py` registrations. Factory functions remain compatibility
wrappers. BF16's compiler-local `_factory` moved to the shared selector.
The large-batch FP8 linear decision moved from the decode recipe to linear's
registrations. The packed linear fallback now imports its own baseline directly.
`units.csv` maps original top-level definitions and each original routing entry;
`coverage.json` verifies targets and records preserved specialization counts.

The new compiler-free forward plan derives requirements from composed calls and
binds the same calls against compiled kernel ABIs. Compilation resolves registered
variants into Tensor source declarations. The new runtime owns scratch/state,
kernels and captured graphs while borrowing caller inputs, outputs and weights.
The three-file residual example exercises the shared safetensors loader without
changes to model-specific compilation or execution code.

Validation includes the complete CPU suite and frozen arithmetic, existing LFM2
request isolation on CUDA, Qwen kernel/speculation checks, and residual-model CUDA
execution at one/four rows with direct and captured execution. CUDA beam search
compiles/checks/times both workloads, persists a target-bound profile and rebuilds
from it. Synthetic scoring tests reject incorrect candidates and small-batch
wins that severely regress larger batches. New-operation discovery, hardware
qualification, ABI argument reordering, borrowed ownership and compiler-free
imports are covered independently.

Only source module/factory ownership changed in the frozen compilation fixture;
all semantic keys, kinds and parameters were checked against the baseline.
Existing numerical kernel bodies remain covered by `tests/test_engine_layout.py`.
No pretrained checkpoint weights were downloaded. Full pretrained-model throughput
and numerical parity were not measured in this migration. The earlier BF16
whole-model baseline compiler failure is not resolved by these ownership changes.
Experimental Hopper overlays and existing specialized state/verification bindings
retain explicit qualification. The new composition path uses static buffer shapes;
it does not claim all operation contracts work on all providers or dynamically
change shapes at runtime.

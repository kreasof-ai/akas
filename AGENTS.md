# AGENTS.md — AKAS

Read this file completely before making changes, running benchmarks, or launching GPU jobs. AKAS is organized with strict structural invariants and forensic benchmark provenance. Most mistakes come from writing redundant scripts or bypassing existing workflows.

---

## 1. Ownership Boundary (`akas` vs `tensor`)

- **`akas` owns:** model declarations (`src/akas/models/`), checkpoint loading/upload (`src/akas/checkpoints/`), quantization formats/codecs (`src/akas/quantization/`), operation families and `@kernel` bodies (`src/akas/ops/`), graph composition/rewrites/policies (`src/akas/compilation/`), autonomous deployment optimization (`src/akas/optimization/`), request state/executor/serving (`src/akas/execution/`, `src/akas/scheduling/`, `src/akas/serving/`), and benchmark history/reporting (`src/akas/benchmarking/`, `benchmarks/`, `tools/`).
- **`tensor` owns:** TileLang/TIRx lowering, CUDA/WebGPU/CPU codegen, `.tbin`/`.tpack` artifact formats, runtime buffer/stream/graph execution, and generic schedule search/timing primitives (`ScheduleSearch`, `ScheduleProfile`, `tune_shapes`, `select_regions`, `cuda_working_set`, `measure_cuda`, `measure_cuda_sequence`).
- **Rule:** If you need a compiler feature (e.g., lowering attributes, buffer views, ABI descriptors), add it as a general compiler feature in `tensor`. Never put model-specific kernels into `tensor`, and never reimplement `tensor`'s compiler or timing primitives inside `akas`.

---

## 2. Do Not Reinvent Existing Workflows

Before writing any script in `tools/` or `build/`, check the existing entry points. **Do not create ad-hoc `tools/<device>_*.py` scripts, parallel `benchmarks/configs/` directories, or custom markdown reports.**

| Task | Existing Canonical Tool / Module | What NOT To Do |
| --- | --- | --- |
| Define workloads | `src/akas/benchmarking/workloads/*.json` (`standard-v1`, `concurrent-prefill-v1`, `single-user-long-context-v1`) | Do not hardcode ad-hoc batch/prefix lists in one-off scripts or mutate versioned workload JSONs. |
| Declare models | `benchmarks/models/<model-id>.json` | Do not add model-name switches to runners or duplicate checkpoint metadata outside `benchmarks/models/`. |
| Run vLLM / SGLang / llama.cpp baselines | `python -m tools.baselines <runtime> <model> [--format ...] [--optimized] [--streaming] [--workload ...]` | Do not write custom baseline wrappers, change baseline flags between AKAS trials, or re-measure baselines on every AKAS tuning run. Measure once, freeze, and lock (`baseline_lock`). |
| Compile & tune a program | `akas compile` / `akas.compilation.compile:compile_checkpoint` (beam search is enabled by default) | Do not set `search=False` and write manual sweep scripts that bypass `akas.compilation.tuning` and independent operation validation. |
| Seed from prior runs | `akas history seed RUN_ID --phase ... --batch ... --chunk ...` or `akas.optimization.incumbents:seed` | Do not manually copy/paste `.tbin` files across incompatible phases or skip requalification of seeded schedules. |
| Optimize a deployment end-to-end | `akas optimize` / `akas.optimization.supervisor:optimize` | Do not promote isolated kernel winners without running full-model evaluation across all requested shapes (`akas.optimization.evaluation`). |
| Run a campaign matrix | `python -m tools.campaign --spec benchmarks/campaigns/<spec>.json --budget ... --slice ... --out build/<dir>` | Do not write custom campaign loops or bypass campaign state/lock management. |
| Profile kernels & phases | `python -m tools.cuda_profile` (`--captured` for CUPTI graph attribution, `--counter-range` for Nsight Compute, `--observe op:arg` for routing/buffers); `tools.vulkan_profile` / `tools.vulkan_tune` | Do not write ad-hoc Python loop timers that measure host dispatch overhead or hot L2 residency without `cuda_working_set`. |
| Render & validate reports | `python -m tools.campaign_report --campaign ...` and `python -m tools.campaign_report --check` | Never hand-edit generated Markdown/CSV files in `benchmarks/reports/` or write custom `REPORT.md` / `FINAL_REPORT.md` files. |
| Archive runs & reports | `python -m tools.benchmark_archive publish`, `check --objects`, `check --tracked`, `prune`, `fetch` | Never commit payload files under `benchmarks/runs/` or `benchmarks/reports/` directly to `main`. |

---

## 3. Structural & Codebase Invariants (Enforced by Tests & CI)

### A. Uniform Six-File Operation & Codec Layout (`tests/test_ops_structure.py`)
Every operation family (`src/akas/ops/<family>/`) and every quantization codec (`src/akas/quantization/<scope>/<format>/`) must have **exact** file parity:
```text
__init__.py        package boundary (codecs export only FORMAT)
definition.py     signatures, OutputABI, VALIDATIONS, policies, IMPLEMENTATIONS
kernel.py         shared portable @kernel computations and inline @T.macro helpers
cuda.py           CUDA specializations, shared bindings, and provider IMPLEMENTATIONS
webgpu.py         WebGPU specializations, shared bindings, and provider IMPLEMENTATIONS
validation.py     independent host/reference numerical and state expectations
README.md         concrete backend specialization reasons and supported contracts
```
- No extra `.py` files or subdirectories are allowed inside any operation or codec package.
- Every registered implementation factory must take `(spec, schedule)` (`tests/test_quantization_conventions.py`).
- Operations must never import from `akas.checkpoints`.

### B. Lazy Compiler Imports (`tests/test_quantization_conventions.py`, `tests/test_engine_layout.py`)
- Importing `akas`, `akas.ops.*`, `akas.quantization.*`, `akas.models.*`, or `akas.execution.*` **must succeed without compiler or framework packages installed**.
- **Never** import `tensor.compiler.*`, `akas.compilation.*`, `tilelang`, `tvm`, `tvm_ffi`, `torch`, or `triton` at module top level in runtime, model, operation, or codec files. Import `tilelang.language as T` and `tensor.compiler.*` **inside** the `(spec, schedule)` factory function.

### C. Functional `@kernel` Boundary (`tests/test_functional_kernels.py`)
- Every standalone device kernel in `src/akas/ops/` and `src/akas/quantization/` must use `@kernel` from `akas.kernels.kernel` with typed `Tensor(shape, dtype)` parameters, `output(shape, dtype)` declarations, and a single unconditional `return`.
- Mutated state inputs must appear in the return value.
- Standalone `@T.prim_func` and `primitive(...)` calls in `ops/` and `quantization/` are prohibited and fail `test_functional_kernels.py`.

### D. Forensic Kernel Hash Lock (`tests/test_engine_layout.py`, `tests/fixtures/kernel-computations.json`)
- All launch bodies and arithmetic `@T.macro` helpers in `ops/` and `quantization/` are AST-hashed and locked in `tests/fixtures/kernel-computations.json` (and inline CUDA source constants in `tests/fixtures/kernel-arithmetic.json`).
- **Do not modify existing kernel bodies in place without qualification.**
  - Prefer adding new schedule/policy parameters or a separate factory while keeping the default computation AST unchanged.
  - If an existing computation body is intentionally replaced, it **must** preserve default bitwise behavior, be verified by a sealed qualification run in `benchmarks/runs/<run-id>/` where `report.json` has `all(check["bitwise"] for check in report["metadata"]["checks"])`, and be registered under `qualified_replacements` in `tests/fixtures/kernel-computations.json`.

---

## 4. Forensic Benchmarking & Qualification Standards

### A. Provenance & Source Integrity (`akas.benchmarking.history.Recorder`)
- Every measurement, qualification, or diagnostic run is recorded through `Recorder`.
- When co-developing with `../tensor`, pass `dependency_sources={"tensor": Path("../tensor")}` so the exact Tensor working tree is snapshotted and checksummed.
- **Never edit source files in `akas` or `tensor` while a measurement or campaign is running.** `Recorder` and `tools.campaign` hash the source tree at start and completion; mid-run edits mark the run `invalidated`.

### B. Timing Protocol Separation
Never mix timing scopes or compute cross-protocol speedups:
1. **`client-stream-burst-v1`**: Concurrent token-arrival timing (`benchmarking.streaming.engine_burst` for AKAS; `tools.baselines --streaming` for vLLM/SGLang). Prefill rate = total prompt tokens / max TTFT; decode rate = remaining 127 tokens / max decode interval.
2. **`native-token-phase-v1` / `native-token-step-v2`**: Synchronized `Executor` phase timings (`akas bench`), excluding HTTP/scheduler queues.
3. **Isolated Kernel / Sequence Diagnostics**: `tools.cuda_profile` (`measure_cuda_sequence` or `--captured` CUPTI) and `tensor.compiler.tuning.measure_cuda` (with `cuda_working_set` to prevent unrealistic hot-L2 small-batch winners).

### C. What "Qualified" Means (Removing `†`)
A profile in `benchmarks/reports/` carries `†` (`unqualified`) until all required checks pass on the **exact measured program binary**:
- **Operation Qualification:** Every operation instance in the compiled phases passes its independent `validation.py` reference check.
- **Workload Qualification:** Every requested batch/prefix shape completes all repetitions with exact token counts, deterministic greedy token IDs across repeats, valid slot reuse, and diagnostic logit/hidden error within tolerance.
- **Independent Full-Model Qualification:** When qualifying a checkpoint/precision path against external baselines (e.g., `LFM2.5-24B-A2B-FP8`), verify three-way greedy agreement (BOS and chat probes) and logit RMS noise floor against vLLM, SGLang, and an FP64/precision-matched oracle (see archived run `20261101113717922004-dcb7262b` and `tests/support/native_fp8_reference.py`).
- **Never** remove `†` by editing report text or weakening tolerances.

### D. Cloud / Remote GPU Discipline (Modal, etc.)
- Use the local GPU (e.g., L40S) for offline compilation, AST/IR inspection, unit tests, and small numerical checks before allocating paid remote GPUs (e.g., H200).
- Verify checkpoint/volume readiness before launching a GPU container.
- Run one GPU benchmark process at a time with explicit timeouts, reuse compiled/qualified caches across continuations, and ensure remote GPU containers are stopped immediately after the run finishes.

---

## 5. Required Verification Before Completing Work

Run these checks locally before declaring any code or report change complete:

```sh
# 1. Lint and format check
uv run --no-sync ruff check src examples
uv run --no-sync ruff format --check src/akas/ops src/akas/kernels src/akas/quantization src/akas/models src/akas/compilation

# 2. Structural, convention, and forensic hash-lock tests
uv run --no-sync python -m pytest \
  tests/test_ops_structure.py \
  tests/test_engine_layout.py \
  tests/test_functional_kernels.py \
  tests/test_quantization_conventions.py \
  tests/test_model_conventions.py \
  tests/test_program_contracts.py \
  tests/test_benchmark_history.py \
  tests/test_benchmark_archive.py \
  tests/test_campaign.py \
  tests/test_campaign_report.py \
  tests/test_optimization.py

# 3. Benchmark report and archive integrity checks
uv run --no-sync python -m tools.campaign_report --check
uv run --no-sync python -m tools.benchmark_archive check --tracked
```

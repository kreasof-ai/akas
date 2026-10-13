# Compilation

`compile.py` turns model graphs into a compiled program. `compile_program()` accepts
phase geometries, request/context bounds and model options. `compile_checkpoint()`
loads weights through the supplied definition; the CLI defaults to this program path.

| File | Responsibility |
| --- | --- |
| `compile.py` | Coordinate phases, state reset, verification commits and publication |
| `selection.py` | Resolve operation implementations and protected schedule parameters |
| `tuning.py` | Numerically qualify candidates, measure workloads and call Tensor beam search |
| `validate.py` | Check each candidate against every call's storage ABI |
| `buffers.py` | Plan scratch reuse from graph lifetimes |
| `state.py` | Lift row-local state addressing into request-slot pools |
| `rewrites.py` | Operation fusion, terminal row pruning, token feedback and deferred state |
| `phases.py` | Describe token phases and proposal-independent speculative controls |
| `emit.py` | Export dependency-tracked sources and invoke Tensor compilation |

Beam search is enabled by default. Candidate axes belong to operations; Tensor owns
the search algorithm and backend compilation. A candidate must preserve numerical
results, immutable inputs, state and buffer layouts before its timing can count.
Qualification is scoped to the compiled workload; it does not imply peak performance
at every other shape.

Each program build writes separate immutable phase/kernel paths. `program.json` is
published last. A failed rebuild leaves the previously published program usable.
Runtime imports do not require the compiler extra or reference frameworks.

```sh
akas compile --definition akas.models.qwen35.model:DenseEngine \
  --checkpoint /path/to/weights --out build/model \
  --config '{"geometries":[[1,1],[1,32]],"max_requests":256,"max_context":4096}'
```

To compile MTP, include `proposal` and `verify` geometries, `round_windows`, and
optionally `deferred`. The checkpoint reader derives its upload branch from declared
weight buffers. Numerical kernels still determine which targets/layouts are supported.

Migration remains in progress: `recipes/`, `factories/` and `transforms/` still serve
unmigrated packed and MoE paths. They must be replaced and qualified before deletion;
they are not extension points for new models.

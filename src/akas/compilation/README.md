# Compilation

`compile.py` turns model graphs into a compiled program. `compile_program()` accepts
phase geometries, request/context bounds and model options. `compile_checkpoint()`
loads weights through the supplied definition; the CLI defaults to this program path.

Graphs are declared with `akas.graph.Graph`: named buffers describe shapes, dtypes
and ownership, and ordered operation calls describe computation. Composition needs
no device or compiler. See the [model guide](../models/README.md) for token models
and the [residual example](../../../examples/graphs/residual/) for a standalone graph:

```python
from akas.compilation.compile import produce
from examples.graphs.residual.config import Config
from examples.graphs.residual.graph import compose

graph = compose(Config(width=256), rows=4)
produce(graph, "build/residual", provider="cuda", target="sm_89")
```

The compiler selects compatible operation implementations and applies graph
optimizations. Operation registrations and rewrite rules live in each operation's
`definition.py`; discovery lives in `akas.kernels.registry`. See the
[operation guide](../ops/README.md) for authoring and the
[execution guide](../execution/README.md) for running a compiled graph with `Forward`.

Schedule profiles can also select operation policies before lowering. Entries
named `policy.<family>` match `phase`, `rows` and `chunk` through Tensor's existing
parameter selector. Their settings override that family's supplied defaults;
ordinary entries continue to select kernels. Effective overrides are recorded
with each compiled phase, including its derived state-only prefill phase.

| File | Responsibility |
| --- | --- |
| `compile.py` | Coordinate phases, state reset, verification commits and publication |
| `selection.py` | Resolve operation implementations and protected schedule parameters |
| `tuning.py` | Numerically qualify candidates, measure workloads and call Tensor beam search |
| `validate.py` | Check each candidate against every call's storage ABI |
| `liveness.py` | Remove output-only work from intermediate prefill chunks |
| `buffers.py` | Plan scratch reuse from graph lifetimes |
| `state.py` | Lift row-local state addressing into request-slot pools |
| `rewrites.py` | Operation fusion, terminal row pruning, token feedback and deferred state |
| `policy.py` | Apply explicit operation-owned physical pipeline choices before search |
| `quantization.py` | Reuse activation quantization across projections |
| `storage.py` | Fold private output casts into producers with typed output contracts |
| `phases.py` | Describe token phases and proposal-independent speculative controls |
| `emit.py` | Export dependency-tracked sources and invoke Tensor compilation |

Request mapping lifts each declared state allocation into the request-slot pool.
Typed views retain their element type: reading byte-backed FP8 storage through an
FP8 view must decode FP8 values after remapping. Views currently cover the whole
allocation and preserve element width; copy regions must stay within one request.

Beam search is enabled by default. Candidate axes belong to operations; Tensor owns
the search algorithm and backend compilation. A candidate must preserve numerical
results, immutable inputs, state and buffer layouts before its timing can count.
Source import scans reuse unchanged content; edits and newly discovered codec
files still update the dependency closure and generated source identity.
Qualification is scoped to the compiled workload; it does not imply peak performance
at every other shape.
Candidate timing binds arguments once and replays a prepared execution plan,
matching the provider path used by serving. Repeated Python argument binding is
excluded. Mutable inputs and outputs are restored outside each timed invocation;
qualification still checks results and state after repeated replays. Traces label
the execution mode so kernel timings can be interpreted with their protocol.

Search keeps one numerical fixture per operation at a time. Candidates sharing
its storage ABI and declared precision reuse it; switching layouts or precision
defaults releases the preceding fixture and device buffers before constructing
another. Matrix tile changes reuse the same fixture. Longer searches therefore
do not accumulate reference weights or KV buffers for each layout. A supplied
schedule starts exploration from its saved geometry and is qualified again.

Each program build writes separate immutable phase/kernel paths. `program.json` is
validated against the loader contract and published last. A failed rebuild leaves the previously published program usable.
Runtime imports do not require the compiler extra or reference frameworks.

```sh
akas compile --definition akas.models.qwen35.model:Qwen35Dense \
  --checkpoint /path/to/weights --out build/model \
  --config '{"geometries":[[1,1],[1,32]],"max_requests":256,"max_context":4096}'
```

To compile MTP, include `proposal` and `verify` geometries, `round_windows`, and
optionally `deferred`. The checkpoint reader derives its upload branch from declared
weight buffers. Numerical kernels still determine which targets/layouts are supported.

Physical pipeline options belong to operation policies, including split projections,
attention partitions, expert layouts, resident operands and recurrent recomputation.
Kernel beam search qualifies schedules inside the selected pipeline. Requested
geometries are compiled explicitly; this is not one shape-polymorphic kernel.

The CUDA FP8 projection policy accepts `minimum_split_groups` with `partitions`:
the latter becomes a partition ceiling, and underfilled launches select the
smallest legal K divisor that reaches the requested group count, or the largest
legal divisor within that ceiling. Wide launches keep one partition. Without
`minimum_split_groups`, `partitions` retains its fixed-count behavior.
`minimum_rows` controls FP8 pipeline eligibility;
`floating_matmul_minimum_rows` independently controls canonical F32 projections.
For native FP8 experts, `moe.fused_quantize_minimum_rows` keeps output quantization
separate below the selected row count when `fused_swiglu=True`. The paired
`expert_swiglu` operation shares the same grouped GEMM body as the fully fused
path, exposing 32-, 64- and 128-column tiles while the separate quantizer retains
its 128-value groups. The default threshold of one preserves full fusion.
FP8 projection schedules also accept `mma_dtype="float16"` to promote the
stored FP8 values exactly before multiplication when tighter accumulation
accuracy is required. Qualify the selected schedules and the resulting model.
Native FP8 expert schedules expose the same operand-promotion choice without
changing checkpoint storage, activation codes, scales or KV precision.

`normalization.fused_residual=True` selects batched residual/RMS fusion for CUDA
reference graphs. It reuses the serial reduction tree and stages row values in
registers, retaining the separate calls' residual and normalization results.
The existing default rewrite policy remains available without this option.

`normalization.fused_staging=True` and `convolution.fused_staging=True` fold a
private FP32-to-FP16 staging cast into the producing operation after linear
staging has been introduced. Computation retains its original precision and the
cast remains at the same output boundary. Public buffers, exposed aliases,
in-place updates and outputs with additional consumers retain separate calls.

`moe.intermediate_dtype="bfloat16"` stores private routed contributions in 16 bits
between expert projection and reduction, retaining FP32 accumulation. The policy
updates the producer, consumer and complete alias family together. Contributions
with other users or external request bindings retain their storage. `float16`
and the default `float32` are also available; qualify the resulting rounding
boundary as part of the model's numerical policy.

`moe.packed=True` stages quantized expert inputs once in aligned expert order.
The paired projection can retain that order for its private output; the down
projection writes contributions in logical token/rank order. Counts, routes,
block scales and accumulation remain unchanged. `packed_minimum_rows` limits
eligibility by workload size. Packing requires a complete mapped CUDA expert
chain with private activation storage and one reduction partition; its cost
belongs in the complete chain's measurement.

`SearchSettings(cache_working_set_bytes=...)` conditions each CUDA measurement
with Tensor's independent working set after restoring mutable outputs and before
timing. The default keeps the existing warm measurements. The footprint and
conditioning artifact identity are recorded in search context, so results from
different residency conditions cannot share a qualification cache entry.

`liveness.py` derives state-only prefill from declared writes and dependencies.
It shares already-qualified kernel artifacts with the normal prefill phase.

For automatic program selection and a ready-to-serve result, use
[the optimization workflow](../optimization/README.md).

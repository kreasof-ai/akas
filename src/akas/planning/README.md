# Composing a model

Use three files per model: `config.py` for geometry, `weights.py` for the chosen
checkpoint reader and weight mapping, `model.py` for operation composition.
The runnable [residual example](../../../examples/models/residual/) follows this
contract. It uses the shared safetensors reader and existing linear/add kernels.

`ForwardPlan` declares named buffers with shape, dtype and ownership role, then
records ordered operation calls. Calls accept positional buffers or named ABI
bindings. Repeated specializations share one kernel declaration. No device or
compiler is needed to compose or serialize a plan.

```python
from akas.compilation.forward import produce
from examples.models.residual.config import Config
from examples.models.residual.model import compose

forward = compose(Config(width=256), rows=4)
produce(forward, "build/residual", provider="cuda", target="sm_89")
```

`compilation.forward` derives requirements from those calls, selects compatible
operation implementations and uses the existing Tensor build pipeline. Pass
`provider="webgpu", target="webgpu-portable-v1"` to select WebGPU implementations.
An unavailable operation/contract/provider fails selection; compilation does not
promise every model or numerical contract on every target.

```python
from akas.execution.forward import Forward

# device, weights, input and output buffers belong to the caller.
with Forward("build/residual", device,
             {**weights, "input": input_buffer, "output": output_buffer},
             graphs=True) as executable:
    executable.run()
```

`Forward` loads checksummed artifacts and binds the same recorded calls. It owns
only omitted scratch/state allocations, kernels and its captured graph. State
starts at zero; caller-supplied state is borrowed. Tensor validates device and
kernel ABI compatibility. Binding validates all buffer shapes and dtypes before
launch. Repeated execution uses the prepared calls or graph and does not select,
compile or search kernels. The compiler is optional on the consuming machine.

An operation's `implementations.py` declares compatible buffer contracts,
providers, phases, optional target/geometry restrictions and supported search
axes. `ops.registry` discovers registrations without importing numerical kernels.
Add hardware implementations next to their operation and qualify their target and
layout explicitly. Equal-priority matches fail as ambiguous. Parameters alone
belong in schedules, rather than another implementation file.

`compilation.tuning.tune` uses Tensor's beam search. Supply representative
`Workload`s and a measurer returning `Measurement(seconds, correct)` after
compiling, checking against a reference, and timing each candidate. Candidates
must pass every workload. The objective minimizes the worst latency ratio against
each workload's default, so a small-batch win cannot hide a large-batch regression.
A finite candidate budget bounds exploration; it cannot guarantee the global best.
The returned `Result.profile(...)` is target-bound and can be passed as
`schedules=` to `compilation.forward.plan` or `produce`. Keep semantic shape and
precision parameters out of the schedule axes. Workload input metadata can carry
ragged lengths, positions and other realistic request conditions.

Existing LFM2/Qwen entry points retain their qualified numerical bindings and
specialized state/verification plans. Their main recipes now share operation
selection; legacy factory entry points delegate to registrations. Experimental
Hopper overlays still require their existing workspace/ABI rewrites and explicit
qualification. They are not automatically interchangeable with ordinary calls.
Forward plans currently describe static buffer geometry; this work does not add
dynamic-shape arithmetic, a universal tokenizer, or new checkpoint encodings.

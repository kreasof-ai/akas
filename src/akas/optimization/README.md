# Optimization

`akas optimize` turns model and workload intent into a qualified serving
deployment. Users supply the checkpoint, workload, budget and optional resource
or latency constraints. The optimizer chooses inference settings.

See [the optimization guide](../../../docs/optimization.md) for commands,
measurement rules, recovery and deployment.

| File | Responsibility |
| --- | --- |
| `context.py` | Validate user intent and discover model declarations |
| `planning.py` | Probe hardware, derive candidate programs and score complete workloads |
| `incumbents.py` | Select compatible schedules from verified historical evidence |
| `supervisor.py` | Persist campaign state, enforce deadlines, isolate failures and publish winners |
| `worker.py` | Compile and qualify one candidate in a disposable process |
| `evaluation.py` | Check model outputs and measure native phases, scheduling and HTTP serving |
| `deployment.py` | Validate and load the selected program without compiling |

Model families declare checkpoint architecture identifiers, supported prefill
search chunks, optional model optimization axes and a speculative head beside
their model declaration. Axes describe implementation choices such as a fused
head or projection split; users do not supply them.
Operation definitions expose applicable pipeline alternatives through
`search_policies(graph, commit, *, provider, target)`. Kernel schedule spaces and
independent numerical qualification remain operation-owned.

New operations inherit compilation and serving orchestration. Add a policy
generator only when the operation has distinct physical pipelines to explore.
Use applicability checks to avoid exposing another backend's algorithms.

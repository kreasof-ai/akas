# Benchmarking

The harness measures any model declaration using the common executor. It records
workloads, timing rules, source identity, correctness evidence and qualified
search results. A new model does not need a benchmark profile module.

Start with [the measurement guide](../../../docs/benchmarking.md) and
[benchmark history](../../../benchmarks/README.md).

| File | Responsibility |
| --- | --- |
| `cases.py` | Request batches, prompt/output lengths and cache depth |
| `session.py` | Common executor adapter |
| `runner.py` | Synchronized warmups, samples and token accounting |
| `report.py` | Reports and matched comparisons |
| `history.py` | Source snapshots, run records, indexes and incumbent schedule exports |
| `replay.py` | Independent numerical evidence |
| `research.py` | Program preparation and token workload execution |

`akas bench` archives results by default. `akas history list` finds earlier runs;
`akas history seed` exports a qualified phase schedule for requalification and
further search. Compiler schedule exploration belongs to
[compilation](../compilation/README.md). Fixed schedules used only by regression
tests live under `tests/support`.

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
| `speculative.py` | Complete block-round measurements and committed-token accounting |
| `runner.py` | Synchronized warmups, samples and token accounting |
| `text.py` | Bounded real-text dataset reading, fixed splits and document crops |
| `report.py` | Reports and matched comparisons |
| `history.py` | Source snapshots, run records, indexes and incumbent schedule exports |
| `archive.py` | Commit-pinned artifact lookup, checksums and on-demand bundle retrieval |
| `replay.py` | Independent numerical evidence |
| `research.py` | Program preparation and token workload execution |

`akas bench` archives results by default. `akas history list` finds earlier runs;
`akas history seed` exports a qualified phase schedule for requalification and
further search. Compiler schedule exploration belongs to
[compilation](../compilation/README.md). Fixed schedules used only by regression
tests live under `tests/support`.

For real-text measurements, pass token lists as `run(..., prompts=token_lists)`.
Their lengths must match the case; the report records their digest and an empty
random seed. Supply `cached_prefixes` too when measuring an existing prefix.
Use exactly the same IDs for the target, speculative decoder and baseline.

`python -m tools.text_workload --checkpoint model.gguf --cache build/datasets
--output build/text-workload.json` prepares a pinned LFM2 MoE workload from
MT-Bench conversation/code/reasoning prompts, GSM8K math and LongBench-v2 documents.
It uses the checkpoint's chat formatter. Calibration and held-out examples are
assigned before runtime tuning; all crops of a document stay in the same split.
The manifest records dataset revisions/checksums, example IDs, token IDs and
preprocessing. Long documents are cropped to 8K and 130944 prompt tokens while
retaining the question and chat delimiters. These modified tasks measure
acceptance and throughput, not original-dataset accuracy. Cache the full datasets
outside benchmark history; retain the manifest and runner as run artifacts.

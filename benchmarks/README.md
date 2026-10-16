# Benchmarks

[Read the guide](../docs/benchmarking.md), browse the
[campaign reports and full comparison tables](reports/README.md), inspect the
[individual run index](RESULTS.md), or query `akas history list`.

- [Models](models/) declare pinned checkpoint sources and baseline options.
- [Workloads](../src/akas/benchmarking/workloads/) define versioned shapes and measurement rules.
- `runs/<UTC>-<id>/run.json` identifies one run, its source, settings and results.
- Each run retains its source snapshot, report and qualified phase schedules.
- `index.json` and `RESULTS.md` are generated from run records.
- `reports/<UTC>-<campaign>/` contains complete campaign tables, baseline data,
  CSV exports, source and runtime settings, and selected kernel schedules.

Publish performance results through these generated records and report bundles.
The [report format](../docs/benchmarking.md#find-a-previous-result) defines the
required files and provenance. CI runs `python -m tools.campaign_report --check`
to verify complete coverage and reject edited tables or arbitrary report pages.

Keep model files, executables and disposable compiler logs in `build/`. Retain
small evidence files beside their run with checksums. Records describe measured
hardware only; an L4 result does not establish L40S performance.

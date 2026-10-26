# Benchmarks

[Read the guide](../docs/benchmarking.md), browse the
[campaign reports and full comparison tables](reports/README.md), inspect the
[individual run index](RESULTS.md), or query `akas history list`.

- [Models](models/) declare pinned checkpoint sources and baseline options.
- [Workloads](../src/akas/benchmarking/workloads/) define versioned shapes and measurement rules.
- [Archive overview](ARCHIVE.md) lists branches and snapshot commits;
  [archive.json](archive.json) maps every archived run and report to its branch,
  full commit hash, path and per-file SHA-256 checksums.
- `runs/<UTC>-<id>/run.json` identifies one local run, its source, settings and results.
- Each run retains its source snapshot, report and qualified phase schedules.
- `index.json` and `RESULTS.md` combine archived summaries with new local records.
- `reports/<UTC>-<campaign>/` contains complete campaign tables, baseline data,
  CSV exports, source and runtime settings, and selected kernel schedules. Archived
  bundles live on `codex/artifacts/<YYYY-MM>-<shard>` branches; links pin commits.

Publish performance results through these generated records and report bundles.
The [report format](../docs/benchmarking.md#find-a-previous-result) defines the
required files and provenance. CI runs `python -m tools.campaign_report --check`
to verify new local reports and generated indexes. Archived historical bundles are
sealed by checksums, including original pages when navigation links were rewritten.

New payload directories are ignored on main. Seal completed runs and reports with:

```sh
python -m tools.benchmark_archive publish
python -m tools.benchmark_archive check --objects
```

Publication validates new reports before sealing them. The initial import of
legacy reports can use `publish --legacy-bootstrap` only while the archive is empty.
CI also rejects payload files tracked on the source branch.

Publish the archive branches named in `archive.json` to origin before committing
the updated catalog and documentation on main. Then remove verified local copies:

```sh
python -m tools.benchmark_archive prune
```

Commits contain at most 192 MiB, branches at most 512 MiB of historical payload.
Every archive commit replaces the previous batch in its tree; each new shard starts
without a parent. A run stays intact. Larger bundles must be reduced or explicitly
handled before publishing. Previous commits remain reachable through their branch.
Never merge archive branches into main, squash their history, force-push them or
delete branches referenced by the catalog.

Listings and documentation need no archive download. To retrieve just one bundle:

```sh
python -m tools.benchmark_archive fetch runs RUN_ID
python -m tools.benchmark_archive fetch reports REPORT_ID
akas history show RUN_ID
```

The fetch command prints its verified location under `build/benchmark-archive/`.
Detailed history reads require the pinned commit locally and explain which branch
to fetch when it is missing. A shallow single-branch clone of main avoids archive
history; a normal full clone still downloads it. Existing main history retains old
payloads until a separately coordinated history rewrite.

Keep model files, executables and disposable compiler logs in `build/`. Retain
small evidence files beside their run with checksums. Records describe measured
hardware only; an L4 result does not establish L40S performance.

A development run (`20261026145843633286-516a8e7e`) was rejected after a compiler pruning error was found. [Correction and rejected-run evidence](https://github.com/kreasof-ai/akas/blob/6243760e7784bcf2e62fe4819696ed64c3d0f62d/benchmarks/runs/20261026152348394023-aff4a4f3/analysis/README.md). Its original record is retained for audit and must not seed performance comparisons. The showcase uses the corrected, four-layer-validated program.

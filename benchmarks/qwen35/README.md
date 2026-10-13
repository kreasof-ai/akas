# Qwen research cases

Model arithmetic lives in AKAS operations. Research files declare geometries and
operation policies and use `akas.benchmarking.research` to compile and run them.
One program owns decode, prefill, verification, proposal and commit phases.

```bash
python -m benchmarks.qwen35.tune --checkpoint /path/to/checkpoint --batches 1 4 16 64 256 --windows 8 --out build/qwen
python -m benchmarks.qwen35.spec_run --prepared build/qwen/program/prepared.json --workload workload.json --window 8
```

`--windows` qualifies speculative widths at compilation. `--window` selects a
qualified width for a workload. Beam search is on by default. Use `--policy` to
change operation pipelines and `--model-options` to change model arithmetic or
storage; both changes require compilation of a new program.

Hopper presets (`modal_hopper_long`, `modal_compact`, `modal_bandwidth`, and other
`modal_*` case files) can run directly on a matching machine. Remote execution
uses one shared entry with the retained CUDA image and persistent volume:

```bash
modal run benchmarks/qwen35/modal_execute.py --checkpoint /cache/models/<revision> --preset modal_hopper_long
```

`modal_bandwidth_micro.py` retains the original independent KV/copy primitive
experiments. Primitive search files still own their standalone candidates; these
results do not replace whole-model qualification. `batch_scaling.py` preserves
nested workload selection and per-slot comparison contracts.

The previous separate `inference.json`, `prefill.json`, Hopper/workspace overlays,
and per-family graph installer flags are replaced by `program.json`, policies,
and `prepared.json` (`akas.research.v1`). Old command flags fail explicitly during
argument parsing. Saved old measurements remain research evidence; old prepared
bundle records must be recompiled. Reports include program/source identities,
actual case dimensions, synchronized timing and `model_throughput_qualified=false`
until a separate correctness comparison establishes that claim.

HTTP replay uses the shared transport:

```bash
akas serve --factory benchmarks.qwen35.server:service --config '{"prepared":"build/qwen/program/prepared.json"}'
```

The shared scheduler preserves request admission, token limits, ragged prompts,
stream counts, cancellation, and resident resources across fresh cohorts.

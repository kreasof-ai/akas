"""Check changed prefill arithmetic against frozen, teacher-forced model logits.

This is a bounded regression gate against a qualified AKAS program, not an
independent model oracle or corpus quality evaluation. Every candidate still
requires independent operation qualification before this gate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from akas.benchmarking.history import digest, write
from akas.benchmarking.session import Session
from akas.models.lfm2.model import LFM2
from tools.context_sweep import vulkan_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[32, 512, 2048, 8192])
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.reference_run.read_text())
    if run["status"] != "passed" or run["metadata"]["checkpoint_sha256"] != digest(args.checkpoint):
        raise ValueError("reference run must be passed and use the same checkpoint")
    cases = {r["prefix_tokens"]: r for r in json.loads(args.prompts.read_text())["cases"]}
    outputs = {r["case"]["prompt"]: r["output_ids"][0] for r in run["results"]}
    if digest(args.prompts) != run["metadata"]["prompt_manifest_sha256"]:
        raise ValueError("changed prompt manifest")
    reference = np.load(args.reference, allow_pickle=False) if args.reference else None
    arrays, rows = {}, []
    passed = True
    checkpoints = {0, 1, 2, 16, 64, 127}
    with vulkan_device() as device:
        with LFM2(args.checkpoint, args.program, device, graphs=True,
                  checkpoint_options={"tokenizer": False}) as model:
            for phase in model.program["phases"]:
                model.batches.select(1, phase["chunk"], phase=phase["phase"])
            with Session(model, batch=1, chunk=args.chunk) as session:
                for prefix in args.prefixes:
                    session.reset()
                    ids = outputs[prefix]
                    actual = [int(session.prefill([cases[prefix]["token_ids"]])[0])]
                    metrics = []
                    for step in range(len(ids)):
                        if step:
                            actual.append(int(session.step([ids[step - 1]])[0]))
                        if step not in checkpoints:
                            continue
                        key = f"p{prefix}_s{step}"
                        values = session.read_feature("logits").reshape(-1)
                        if not np.isfinite(values).all():
                            raise ValueError("nonfinite logits")
                        arrays[key] = values.copy()
                        if reference is not None:
                            expected = reference[key].astype("float64")
                            delta = values.astype("float64") - expected
                            relative = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30))
                            cosine = float(np.dot(values, expected) / max(
                                np.linalg.norm(values.astype("float64")) * np.linalg.norm(expected), 1e-30))
                            ok = relative < .01 and cosine > .9999
                            passed &= ok
                            metrics.append(dict(step=step, relative_rms=relative, cosine=cosine, passed=ok))
                    exact = actual == ids
                    passed &= exact
                    row = dict(prefix=prefix, output_count=len(actual), matching_argmax=sum(
                        a == b for a, b in zip(actual, ids)), exact_teacher_forced_predictions=exact,
                        sampled_logits=metrics)
                    rows.append(row)
                    print(prefix, "matches", row["matching_argmax"], "/", len(ids),
                          "max_rms", max((m["relative_rms"] for m in metrics), default=0), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    write(args.out, dict(status="passed" if passed else "failed", scope=__doc__,
        program_sha256=digest(args.program / "program.json"), reference_run_id=run["id"],
        reference_sha256=digest(args.reference) if args.reference else None,
        prompt_manifest_sha256=digest(args.prompts), checkpoint_sha256=digest(args.checkpoint),
        logits_sha256=digest(args.out.with_suffix(".npz")), relative_rms_limit=.01,
        cosine_limit=.9999, rows=rows))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

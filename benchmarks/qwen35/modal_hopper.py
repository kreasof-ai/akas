"""Qualify Hopper matrix instructions before the unchanged full C8 replay."""

from akas.compilation.paths import source_path
from akas.compilation.paths import tensor_source

from benchmarks.qwen35.modal_compact import app, image, volume, measure_compact


@app.function(
    image=image,
    cpu=16,
    memory=98304,
    timeout=1800,
    volumes={"/cache": volume},
    scaledown_window=2,
)
def prepare_hopper(
    prepared,
    expert_mma="control",
    hopper_attention=False,
    expert_columns=128,
    attention_load_rows=32,
):
    import hashlib
    import json
    import os
    from pathlib import Path
    from akas.compilation.transforms.hopper import produce

    os.chdir("/workspace")
    volume.reload()
    from akas.artifacts.fingerprints import block_fp8_hashes as implementation_hashes

    decoder = source_path(prepared["base"]["paths"]["decoder"]) / "inference.json"
    if json.loads(decoder.read_text())["implementation"] != implementation_hashes():
        # Regenerate through the producers; never relabel frozen bundle hashes.
        from benchmarks.qwen35.modal_h200 import prepare
        from benchmarks.qwen35.modal_compact import prepare_compact

        print("Regenerating control bundles for the new compiler/runtime", flush=True)
        prepared = prepare_compact.local(prepare.local())
    sources = [
        source_path("src/tensor/compiler/build.py"),
        source_path("src/tensor/runtime/cuda_target.py"),
        source_path("src/akas/compilation/transforms/hopper.py"),
        source_path('src/akas/ops/moe/cuda/hopper.py'),
        source_path('src/akas/ops/linear/cuda/hopper.py'),
        source_path('src/akas/ops/attention/cuda/hopper_prefill.py'),
        source_path('src/akas/artifacts/fingerprints.py'),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    control = source_path(prepared["prefill"]) / "prefill.json"
    identity = hashlib.sha256(
        json.dumps(
            dict(
                hashes=hashes,
                expert_mma=expert_mma,
                hopper_attention=hopper_attention,
                expert_columns=expert_columns,
                attention_load_rows=attention_load_rows,
                control_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    bundle = produce(
        prepared["prefill"],
        source_path("/cache/hopper") / identity,
        expert_mma=expert_mma,
        hopper_attention=hopper_attention,
        expert_columns=expert_columns,
        attention_load_rows=attention_load_rows,
    )
    result = dict(
        prepared,
        hopper=str(bundle),
        hopper_source_hashes=hashes,
        source_identity=identity,
    )
    (bundle / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def hopper_main(
    prepared_file: str,
    out: str = "build/qwen35-h200-hopper",
    expert_mma: str = "control",
    verification_prepared_file: str = "",
    hopper_attention: bool = False,
    expert_columns: int = 128,
    attention_load_rows: int = 32,
):
    from datetime import datetime, timezone
    import json
    from pathlib import Path

    root = source_path(out)
    root.mkdir(parents=True, exist_ok=True)
    if expert_mma not in (
        "control",
        "fp8",
        "bf16",
        "bf16-experts",
        "mma-experts",
        "bf16-32",
        "bf16-pairs",
        "bf16-async",
        "bf16-pipeline",
    ):
        raise ValueError("invalid expert MMA variant")
    prepared = prepare_hopper.remote(
        json.loads(source_path(prepared_file).read_text()),
        expert_mma,
        hopper_attention,
        expert_columns,
        attention_load_rows,
    )
    if verification_prepared_file:
        verification = json.loads(source_path(verification_prepared_file).read_text())
        for name in ("checkpoint", "model_revision", "target"):
            if prepared["base"][name] != verification["base"][name]:
                raise ValueError("verification control mismatch")
        prepared = dict(
            prepared,
            base=dict(
                prepared["base"],
                paths=dict(
                    prepared["base"]["paths"],
                    **{
                        k: verification["base"]["paths"][k]
                        for k in ("verify", "repair")
                    },
                ),
            ),
            verification_chunk=verification["verification_chunk"],
            verification_source_identity=verification["source_identity"],
            long_source_hashes=verification.get("long_source_hashes", {}),
        )
    (root / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + prepared["source_identity"]
    )
    (root / "run.json").write_text(
        json.dumps(dict(volume="tensor-qwen35-h200", run_id=run_id), indent=2) + "\n"
    )
    result = measure_compact.remote(prepared, run_id)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Retained Hopper replay: tensor-qwen35-h200 /hopper-runs/" + run_id)

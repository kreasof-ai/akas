"""Produce an experimental MTP draft bundle from selected native target kernels."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from akas.compilation.build import build
from akas.compilation.plan import Bundle
from akas.compilation.plan import Kernel
from akas.compilation.plan import Source
from akas.compilation.bundles import copy
from akas.execution.bindings.block_fp8.geometry import validate
from akas.artifacts.kernels import identity
from akas.artifacts.fingerprints import (
    block_fp8_hashes as mtp_implementation_hashes,
)


def produce(checkpoint, decoder, out):
    validate(checkpoint.config)
    decoder, out = Path(decoder).resolve(), Path(out).resolve()
    if out == decoder or out.is_relative_to(decoder):
        raise ValueError("MTP output must be separate from the target bundle")
    manifest = json.loads((decoder / "inference.json").read_text())
    if manifest["schema"] != "tensor.qwen35-batch.v1" or manifest[
        "config"
    ] != json.loads(json.dumps(asdict(checkpoint.config))):
        raise ValueError("target bundle does not match the checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    (out / "inference.json").unlink(missing_ok=True)
    lineage = dict(
        path=str(decoder),
        manifest_sha256=hashlib.sha256(
            (decoder / "inference.json").read_bytes()
        ).hexdigest(),
        implementation=manifest["implementation"],
    )
    manifest = copy(decoder, out, "inference.json")
    kernels = []
    r = manifest["slots"]
    for kind, p, module, factory, args in (
        (
            "mtp_join",
            dict(r=r, c=checkpoint.config.width, eps=checkpoint.config.epsilon),
            "akas.ops.mtp.cuda",
            "make_kernel",
            None,
        ),
        (
            "mtp_cast",
            dict(r=r, c=checkpoint.config.width),
            "akas.ops.mtp.cuda",
            "make_kernel",
            None,
        ),
        (
            "bf16_linear",
            dict(r=r, k=2 * checkpoint.config.width, o=checkpoint.config.width),
            "akas.ops.vocabulary.cuda.bf16",
            "bf16_head_kernel",
            None,
        ),
    ):
        key = identity(kind, p)
        args = (
            (kind, p)
            if factory == "make_kernel"
            else (dict(p, columns=128, depth=128),)
        )
        kernels.append(Kernel(key, kind, p, Source(module, factory, args)))
    manifest.update(
        draft="qwen35-mtp",
        target_config=asdict(checkpoint.config),
        config=asdict(replace(checkpoint.config, layers=("full_attention",))),
        implementation=mtp_implementation_hashes(),
        target_decoder_lineage=lineage,
        mtp_weight_bytes=checkpoint.weight_bytes,
        speculative_verification=False,
        model_throughput_qualified=False,
        full_stress_target_reached=False,
    )
    build(Bundle(manifest, "inference.json", kernels), out)
    return out

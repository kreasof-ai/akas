"""Compile an explicit experimental Mali schedule without changing defaults."""

import argparse
import json
from pathlib import Path
import shutil

from akas.compilation import compile_checkpoint
from akas.models.lfm3 import LFM3
from tensor.compiler.search import ScheduleProfile


def package(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("runtime destination must be separate from the compiler directory")
    seen = set()

    def scan(value):
        if isinstance(value, dict):
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
        elif isinstance(value, str) and value.endswith((".json", ".tbin")):
            visit(value)

    def visit(name):
        path = source / name
        if not path.resolve().is_relative_to(source):
            raise ValueError("dependency escapes the compiled program")
        if name in seen or not path.is_file():
            return
        seen.add(name)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if path.suffix == ".json":
            scan(json.loads(path.read_text()))

    visit("program.json")
    if not seen:
        raise ValueError("compiled program.json is missing")
    print(f"Packaged {len(seen)} runtime files in {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-out", type=Path)
    parser.add_argument("--chunk", type=int, choices=[128, 256, 512], default=128)
    parser.add_argument("--half-layout", type=int, choices=range(8), default=6)
    parser.add_argument("--half-k", type=int, choices=[16, 32, 64], default=64)
    parser.add_argument("--decode-layout", type=int, choices=range(8), default=2)
    parser.add_argument("--stage-weights", action="store_true")
    args = parser.parse_args()
    entries = []
    for kind in ("linear", "ffn", "linear_add"):
        for encoding in (2, 14):
            entries.append(dict(operation=kind, parameters=dict(r=1, ggml=True, type=encoding),
                                schedule=dict(implementation="linear.webgpu.packed:vector-decode",
                                              decode_layout=args.decode_layout)))
        for rows in (32, args.chunk):
            entries.append(dict(operation=kind, parameters=dict(r=rows, ggml=True, half_accum=True),
                                schedule=dict(implementation="linear.webgpu.short_half",
                                              half_layout=args.half_layout, half_k=args.half_k)))
    profile = ScheduleProfile(dict(schema="tensor.schedule-profile.v1", provider="webgpu",
                                   target="webgpu-portable-v1", entries=entries))
    options = dict(prefill_chunk=args.chunk, weight_arena=False, activation_quantization="half",
                   fused_sandwich=True, q4_layout="raw", decode_splits=32, sliding_splits=4,
                   ring_power_of_two=True, fused_rotary_norm=True, fused_attention_gate=True)
    policy = dict(linear=dict(stage_weights=True, stage_inputs=True)) if args.stage_weights else None
    compile_checkpoint(LFM3, args.checkpoint, args.out, checkpoint_options=dict(tokenizer=False),
                       geometries=[(1, 1), (1, 32), (1, args.chunk)], max_requests=1,
                       max_context=8320, model_options=options, provider="webgpu",
                       target="webgpu-portable-v1", search=False, schedules=profile, policy=policy)
    (args.out / "android-schedule.json").write_text(json.dumps(profile.data, indent=2) + "\n")
    if args.runtime_out:
        package(args.out, args.runtime_out)


if __name__ == "__main__":
    main()

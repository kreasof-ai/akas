"""Compile the retained LFM3 configuration using its qualified seed schedules.

Place seed-profile-final.json beside this script; use the pinned checkpoint in
benchmarks/models/lfm3-1.7b.json. CUDA Toolkit 12.9 and NVIDIA L4 were measured.
"""
import json
from pathlib import Path
from tensor.compiler.search import ScheduleProfile
from akas.compilation import compile_checkpoint
from akas.compilation.tuning import SearchSettings
from akas.models.lfm3 import LFM3
from tools.lfm3_benchmark import PROFILE

compile_checkpoint(
    LFM3, PROFILE.checkpoint, PROFILE.program,
    checkpoint_options={"tokenizer": False},
    geometries=[(1, 1), (1, 32), (1, 512), (1, 1024)],
    max_requests=1, max_context=131072,
    model_options={"prefill_chunk": 1024, **PROFILE.optimized_options},
    schedules=ScheduleProfile(json.loads(Path(__file__).with_name("seed-profile-final.json").read_text())),
    search_settings=SearchSettings(budget=1, samples=2, repeats=3),
)

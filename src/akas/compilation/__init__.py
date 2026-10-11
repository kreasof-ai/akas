"""Optional offline compilation driven by model definitions and execution profiles."""

from importlib import import_module

PROFILES = {
    "packed": ("akas.compilation.profiles.packed", "produce"),
    "batched": ("akas.compilation.profiles.batched", "produce"),
    "batched-bank": ("akas.compilation.profiles.batched", "produce_bank"),
    "hybrid.decode": ("akas.compilation.profiles.hybrid.decode", "produce"),
    "hybrid.prefill": ("akas.compilation.profiles.hybrid.prefill", "produce"),
    "hybrid.draft": ("akas.compilation.profiles.hybrid.draft", "produce"),
    "hybrid.draft-prefill": (
        "akas.compilation.profiles.hybrid.draft_prefill",
        "produce",
    ),
}


def compile_checkpoint(
    definition, checkpoint, out, *, profile, checkpoint_options=None, **options
):
    """Load with a supplied definition; reusable profiles never name a model family."""
    module, factory = PROFILES[profile]
    loaded = definition.checkpoint_type(checkpoint, **(checkpoint_options or {}))
    return getattr(import_module(module), factory)(loaded, out, **options)

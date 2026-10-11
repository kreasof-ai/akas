"""Optional offline compilation driven by model definitions and execution profiles."""

from importlib import import_module

PROFILES = {
    "packed": ("akas.compilation.recipes.packed", "produce"),
    "batched": ("akas.compilation.recipes.bf16", "produce"),
    "batched-bank": ("akas.compilation.recipes.bf16", "produce_bank"),
    "hybrid.decode": ("akas.compilation.recipes.block_fp8.decode", "produce"),
    "hybrid.prefill": ("akas.compilation.recipes.block_fp8.prefill", "produce"),
    "hybrid.draft": ("akas.compilation.recipes.block_fp8.draft", "produce"),
    "hybrid.draft-prefill": (
        "akas.compilation.recipes.block_fp8.draft_prefill",
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

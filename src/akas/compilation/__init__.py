"""Optional offline compilation driven by model definitions and execution profiles."""

from importlib import import_module

PROFILES = {
    "program": ("akas.compilation.compile", "compile_program"),
    "reference": ("akas.compilation.compile", "compile_model"),
    "floating": ("akas.compilation.compile", "compile_model"),

}


def compile_checkpoint(
    definition, checkpoint, out, *, profile="program", checkpoint_options=None, **options
):
    """Load with a supplied definition; reusable profiles never name a model family."""
    module, factory = PROFILES[profile]
    loaded = definition.checkpoint_type(checkpoint, **(checkpoint_options or {}))
    if profile == "program":
        options["definition"] = definition.compose
        if getattr(definition, "model_defaults", None):
            options["model_options"] = {**definition.model_defaults, **options.get("model_options", {})}
    if profile in ("reference", "floating"):
        options["definition"] = definition.definition
        if profile == "floating":
            options.setdefault(
                "prefill_chunk",
                min(
                    options.get("capacity", 4096),
                    128 if options.get("provider", "cuda") == "cuda" else 32,
                ),
            )
    return getattr(import_module(module), factory)(loaded, out, **options)

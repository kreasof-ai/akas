"""Optional offline compilation driven by model definitions and execution profiles."""

from importlib import import_module

PROFILES = {
    "program": ("akas.compilation.compile", "compile_program"),
    "reference": ("akas.compilation.compile", "compile_program"),
    "floating": ("akas.compilation.compile", "compile_program"),
}


def compile_checkpoint(
    definition,
    checkpoint,
    out,
    *,
    profile="program",
    checkpoint_options=None,
    **options,
):
    """Load with a supplied definition; reusable profiles never name a model family."""
    module, factory = PROFILES[profile]
    loaded = definition.checkpoint_type(checkpoint, **(checkpoint_options or {}))
    if profile in ("program", "reference", "floating"):
        options["definition"] = definition.compose
        if getattr(definition, "model_defaults", None):
            options["model_options"] = {
                **definition.model_defaults,
                **options.get("model_options", {}),
            }
    if profile in ("reference", "floating"):
        rows = options.pop("batch", 1)
        context = options.pop("capacity", 4096)
        default_chunk = min(
            context, 128 if options.get("provider", "cuda") == "cuda" else 32
        )
        chunk = options.pop(
            "prefill_chunk", default_chunk if profile == "floating" else 1
        )
        geometries = [(rows, 1)]
        if chunk > 1:
            geometries.append((rows, chunk))
        options.setdefault("geometries", geometries)
        options.setdefault("max_requests", rows)
        options.setdefault("max_context", context)
        model_options = options.setdefault("model_options", {})
        for name in ("kv_dtype", "max_weight_bytes"):
            if name in options:
                model_options[name] = options.pop(name)
    return getattr(import_module(module), factory)(loaded, out, **options)

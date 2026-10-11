"""Explicit dimensions supported by the current fixed-layout FP8 kernels."""

FIELDS = dict(
    width=2048,
    heads=16,
    kv_heads=2,
    head_dim=256,
    rotary_dim=64,
    key_heads=16,
    value_heads=32,
    key_dim=128,
    value_dim=128,
    conv_width=4,
    experts=256,
    top_k=8,
    expert_width=512,
)


def validate(config):
    for name, value in FIELDS.items():
        if getattr(config, name, None) != value:
            raise ValueError(f"hybrid FP8 execution profile requires {name}={value}")


def metadata(config):
    return {name: getattr(config, name) for name in (*FIELDS, "vocab")}


def from_manifest(manifest):
    """Read declared dimensions; legacy bundles identify the same fixed profile."""
    if "geometry" in manifest:
        return manifest["geometry"]
    geometry = {
        key: manifest.get("config", {}).get(key, value) for key, value in FIELDS.items()
    }
    vocab = manifest.get("config", {}).get("vocab")
    if vocab is None:
        vocab = next(
            (
                row["parameters"].get("vocab")
                for row in manifest["kernels"].values()
                if row.get("kind") in ("embedding", "argmax")
            ),
            None,
        )
    if vocab is None:
        raise ValueError("bundle does not declare vocabulary geometry")
    return {**geometry, "vocab": vocab}

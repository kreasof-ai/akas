"""Strict checkpoint metadata and compressed-tensors declarations."""

import json


def block_fp8_config(config):
    """Validate the shared compressed-tensors block-FP8 checkpoint contract."""
    q = config.get("quantization_config")
    if q is None:
        return False
    groups = q.get("config_groups", {})
    if (
        q.get("quant_method") != "compressed-tensors"
        or q.get("format") != "float-quantized"
        or len(groups) != 1
        or q.get("kv_cache_scheme") is not None
    ):
        raise ValueError("unsupported block-FP8 checkpoint quantization")
    group = next(iter(groups.values()))
    weights, activations = group.get("weights", {}), group.get("input_activations", {})
    if (
        weights.get("num_bits") != 8
        or weights.get("type") != "float"
        or weights.get("strategy") != "block"
        or weights.get("block_structure") != [128, 128]
        or weights.get("dynamic") is not False
        or weights.get("symmetric") is not True
        or activations.get("num_bits") != 8
        or activations.get("type") != "float"
        or activations.get("strategy") != "group"
        or activations.get("group_size") != 128
        or activations.get("dynamic") is not True
        or activations.get("symmetric") is not True
    ):
        raise ValueError(
            "FP8 requires static 128x128 weights and dynamic group-128 activations"
        )
    return True


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def json_document(value):
    return json.loads(value, object_pairs_hook=_object)

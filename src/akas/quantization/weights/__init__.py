"""Weight storage and numerical codecs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Weight:
    """Logical weight geometry and its independently encoded physical pieces.

    Concatenation and group permutations describe checkpoint layout, not a
    materialized decoded matrix. Consumers bind each piece's format reader.
    """

    shape: tuple[int, ...]
    buffers: tuple
    parts: tuple[dict, ...]

    def __post_init__(self):
        from math import prod

        if not 1 <= len(self.parts) <= 4 or len(self.buffers) != len(self.parts):
            raise ValueError("a weight requires one to four encoded pieces")
        if len(self.shape) not in (2, 3) or any(dim <= 0 for dim in self.shape):
            raise ValueError("a weight requires positive matrix or expert geometry")
        if sum(part["rows"] for part in self.parts) != prod(self.shape[:-1]):
            raise ValueError("weight pieces do not cover the logical rows")


def bind_parts(parts, depth, *, provider):
    """Compose format readers for a logical concatenation of encoded matrices."""
    import tilelang.language as T
    from akas.quantization import Codec, Storage
    from akas.quantization._shared.access import matrix_tile, permuted_reader

    codecs = tuple(
        bind(
            {"weight_format": part["format"], "weight_options": part["options"]},
            (part["rows"], depth),
            provider=provider,
        )
        for part in parts
    )
    if any(codec.storage.metadata for codec in codecs):
        raise ValueError("segmented weights currently require embedded metadata")
    readers = tuple(
        permuted_reader(codec.read, part.get("row_order"), part.get("column_order"))
        for codec, part in zip(codecs, parts)
    )
    bounds = [0]
    for part in parts:
        bounds.append(bounds[-1] + part["rows"])
    count = len(parts)
    if not 1 <= count <= 4:
        raise ValueError("segmented weights require one to four pieces")

    @T.macro
    def read(encoded, row, column):
        value = T.alloc_var("float32")
        safe_column = T.max(0, T.min(column, depth - 1))
        if row < bounds[1]:
            value = readers[0]((encoded[0],), row, safe_column)
        else:
            if count > 1:
                if row < bounds[2]:
                    value = readers[1]((encoded[1],), row - bounds[1], safe_column)
                else:
                    if count > 2:
                        if row < bounds[3]:
                            value = readers[2](
                                (encoded[2],), row - bounds[2], safe_column
                            )
                        else:
                            if count > 3:
                                if row < bounds[4]:
                                    value = readers[3](
                                        (encoded[3],), row - bounds[3], safe_column
                                    )
                                else:
                                    value = 0
                            else:
                                value = 0
                    else:
                        value = 0
            else:
                value = 0
        return value

    storage = Storage(
        codecs[0].storage.values,
        tuple(codec.storage.values for codec in codecs[1:]),
        tuple(f"part{index}" for index in range(1, count)),
    )
    direct = (
        codecs[0]
        if count == 1
        and not (parts[0].get("row_order") or parts[0].get("column_order"))
        else None
    )
    return Codec(
        storage,
        read,
        direct.read_tile if direct else matrix_tile(read, (bounds[-1], depth)),
        read_pair=direct.read_pair if direct else None,
        dot=direct.dot if direct else None,
        dot_half=direct.dot_half if direct else None,
        dot_float=direct.dot_float if direct else None,
        stage_tile=direct.stage_tile if direct else None,
    )


def supports_parts(parts, depth, capability, *, provider):
    """Packed access requires one unpermuted encoded piece, with format support."""
    from akas.quantization.formats import get

    if len(parts) != 1 or parts[0].get("row_order") or parts[0].get("column_order"):
        return False
    part = parts[0]
    return get("weights", part["format"]).supports(
        capability, (part["rows"], depth), provider=provider, **part["options"]
    )


def bind(parameters, shape, *, provider):
    """Resolve a weight reader before constructing an operation's device body.

    Legacy graphs identify GGML weights by their numeric ``type`` parameter.
    Keep that translation here so operations never interpret encoding IDs.
    New graphs may declare ``weight_format`` and ``weight_options`` explicitly.
    """
    from akas.quantization.formats import get

    name = parameters.get("weight_format", "ggml")
    options = dict(parameters.get("weight_options", {}))
    if name == "ggml" and "type" in parameters:
        options.setdefault("encoding", parameters["type"])
    return get("weights", name).bind(tuple(shape), provider=provider, **options)

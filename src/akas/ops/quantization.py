"""Track quantized scratch validity while composing an ordered execution plan."""


class ActivationCache:
    """Borrow source/output buffers; reuse quantization until a source is written."""

    def __init__(self, sources, quantized, *, rows):
        self.sources = sources
        self.quantized = quantized
        self.rows = rows
        self.valid = set()

    def get(self, name, emit):
        if name not in self.valid:
            source = self.sources[name]
            parameters = dict(r=self.rows, k=source.shape[-1])
            if len(source.shape) == 3:
                parameters["top"] = source.shape[-2]
            bits, scales = self.quantized[name]
            emit("quantize", parameters, x=source, out=bits, scales=scales)
            self.valid.add(name)
        return self.quantized[name]

    def invalidate(self, buffer):
        self.valid.difference_update(
            name for name in tuple(self.valid) if self.sources[name] is buffer
        )

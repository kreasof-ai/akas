"""Serve the same compiled research program through AKAS's shared HTTP transport.

Use `akas serve --factory benchmarks.qwen35.server:service --config
'{"prepared":"build/research/program/prepared.json"}'`. Separate prefill,
draft, repair and workspace bundle flags have been replaced by program policies.
"""

from contextlib import contextmanager
import json
from pathlib import Path


@contextmanager
def service(*, prepared, window=None, capture=False):
    import tensor
    from akas.benchmarking.research import definition, validate
    from akas.config import EngineConfig
    from akas.engine import Engine
    from akas.generation.speculative.decoding import SpeculativeDecoder
    from akas.models.heads.mtp import MTPHead
    from akas.serving import Service
    from akas.text.tokenizer import Tokenizer

    settings = validate(json.loads(Path(prepared).read_text()))
    model = definition(settings["definition"])
    with tensor.Device(provider=settings["provider"]) as device:
        with model(settings["checkpoint"], settings["program"], device,
                   checkpoint_options=settings["checkpoint_options"], **settings["model_options"]) as executor:
            backend = executor
            if window is not None:
                if window not in settings["windows"]:
                    raise ValueError("server window was not compiled")
                backend = SpeculativeDecoder(executor, MTPHead(executor), window=window, capture=capture)
            tokenizer = Tokenizer.from_pretrained(settings["checkpoint"])
            with Engine(backend, config=EngineConfig(prefill_chunk=settings["chunk"], prefill_requests=executor.capacity)) as engine:
                yield Service(engine, encode=tokenizer.encode,
                              metadata=dict(implementation="akas.program", source_identity=settings["source_identity"],
                                            program_sha256=settings["program_sha256"], policy=settings["policy"],
                                            slots=executor.capacity, context=executor.context))


if __name__ == "__main__":
    raise SystemExit("Use akas serve --factory benchmarks.qwen35.server:service --config '{\"prepared\":\"...\"}'")

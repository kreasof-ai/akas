"""Caller-side model setup for the shared HTTP server.

Run from the checkout: python -m akas serve --factory examples.serving:lfm2
--config '{"checkpoint":"model.gguf","bundle":"build/lfm2"}'
"""

from contextlib import contextmanager
from akas import Engine, EngineConfig
from akas.serving import Service


@contextmanager
def lfm2(
    *,
    checkpoint,
    bundle,
    capacity=1,
    context=None,
    provider="cuda",
    device=0,
    graphs=True,
):
    import tensor
    from akas.models.lfm2.model import LFM2

    with tensor.Device(device, provider=provider) as gpu:
        with Engine.from_checkpoint(
            LFM2,
            checkpoint,
            bundle,
            gpu,
            capacity=capacity,
            context=context,
            graphs=graphs,
        ) as engine:
            tokenizer = engine.model.checkpoint.tokenizer
            yield Service(
                engine,
                lambda text: tokenizer.encode(text, add_bos=False),
                {
                    "model": str(checkpoint),
                    "provider": provider,
                    "prefix_cache": False,
                    "quality_qualified": False,
                },
            )


@contextmanager
def qwen35_dense(
    *,
    checkpoint,
    bundle,
    capacity=256,
    context=4096,
    mtp=False,
    single_graph=False,
    deferred=False,
    fused_head=False,
    verification_window=4,
):
    import tensor
    from akas.text.tokenizer import Tokenizer
    from akas.models.qwen35.model import DenseEngine

    tokenizer = Tokenizer.from_pretrained(checkpoint)
    with tensor.Device() as device:
        target = DenseEngine(
            checkpoint,
            bundle,
            device,
            capacity=capacity,
            context=context,
            fused_head=fused_head,
        )
        model = target
        engine = None
        try:
            if mtp:
                from akas.generation.speculative.decoding import SpeculativeDecoder
                from akas.models.heads.mtp import MTPHead

                model = SpeculativeDecoder(
                    target, MTPHead(target),
                    window=verification_window,
                    capture=single_graph,
                    deferred=deferred,
                )
            engine = Engine(model, config=EngineConfig())
            with engine:
                for count in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                    if count <= capacity:
                        target.batch(count)
                        if mtp:
                            target.batch(count, verification_window, verify=True)
                            model.proposer.frame(count, 1)
                            model.proposer.frame(count, verification_window)
                    if count <= min(capacity, 32):
                        target.batch(count, 32)
                        if mtp:
                            model.proposer.frame(count, 32)
                yield Service(
                    engine,
                    lambda text: tokenizer.encode(
                        text, add_bos=False, parse_special=True
                    ),
                    {
                        "model": str(checkpoint),
                        "provider": "cuda",
                        "weights": "BF16",
                        "kv_dtype": "bfloat16",
                        "state_dtype": "float32",
                        "prefix_cache": False,
                        "program": target.program["schema"],
                        "quality_qualified": False,
                    },
                )
        finally:
            if engine is None:
                model.close()

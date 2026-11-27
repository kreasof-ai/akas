"""Measure a compiled CUDA text program through the shared stream collector."""

import asyncio

from akas.benchmarking.streaming import engine_burst
from tools.baselines.common import main, measurements, synthetic_prompts


def run(config):
    import tensor

    from akas.artifacts.read import read_program
    from akas.config import EngineConfig
    from akas.engine import Engine
    from akas.optimization.context import discover, symbol

    bundle = config["bundle"]
    program = read_program(bundle)
    if (
        max(prefix for _, prefix in config["shapes"]) + config["output_tokens"]
        > program["max_context"]
    ):
        raise ValueError("compiled context is too small for the workload")
    checkpoint = config["checkpoint"]
    definition = symbol(discover(checkpoint))
    prefill = [phase for phase in program["phases"] if phase["phase"] == "prefill"]
    chunk = max((phase["chunk"] for phase in prefill), default=1)
    group = max(
        (phase["rows"] for phase in prefill if phase["chunk"] == chunk), default=1
    )
    engine_config = EngineConfig(prefill_chunk=chunk, prefill_requests=group)
    settings = dict(
        bundle=bundle,
        max_requests=program["max_requests"],
        max_context=program["max_context"],
        prefill_chunk=chunk,
        prefill_requests=group,
        graphs=True,
    )
    report = dict(runtime="akas", settings=settings, cases=[])
    with (
        tensor.Device(provider="cuda") as device,
        Engine.from_checkpoint(
            definition,
            checkpoint,
            bundle,
            device,
            config=engine_config,
            checkpoint_options={"tokenizer": False},
        ) as engine,
    ):

        def invoke(batch, prefix):
            return asyncio.run(
                engine_burst(
                    engine,
                    synthetic_prompts(config["vocab"], batch, prefix),
                    config["output_tokens"],
                )
            )

        measurements(config, invoke, report)
    return report


if __name__ == "__main__":
    main(run)

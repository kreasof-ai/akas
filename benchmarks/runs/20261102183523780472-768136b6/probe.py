"""SGLang eager preflight using the common worker protocol."""

from tools.baselines.common import main, synthetic_prompts, measurements, phase_metrics, stream_burst


def run(config):
    import time
    import torch
    import sglang

    settings = dict(
        model_path=config["checkpoint"],
        dtype="bfloat16",
        context_length=max(p for _, p in config["shapes"])
        + config["output_tokens"]
        + 8,
        max_running_requests=max(b for b, _ in config["shapes"]),
        chunked_prefill_size=2048,
        mem_fraction_static=0.85 if config.get("optimized") else 0.45,
        disable_cuda_graph=not config.get("optimized", False),
        enable_metrics=True,
        disable_radix_cache=True,
        random_seed=17,
        log_level="info",
    )
    settings.update(config["runtime_options"])
    if config.get("optimized"):
        settings["cuda_graph_bs_decode"] = sorted({b for b, _ in config["shapes"]})
    if config.get("streaming"):
        settings.update(skip_tokenizer_init=True, stream_interval=1,
                        incremental_streaming_output=False)
    if config["mode"] == "mtp":
        settings.update(
            speculative_algorithm="EAGLE",
            speculative_num_steps=1,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=2,
        )
    sampling = dict(
        temperature=0, max_new_tokens=config["output_tokens"], ignore_eos=True
    )
    report = dict(
        runtime="sglang",
        version=sglang.__version__,
        settings=settings,
        sampling=sampling,
        cases=[],
    )
    engine = sglang.Engine(**settings)
    try:
        if config.get("streaming"):
            async def generate(index, prompt):
                stream = await engine.async_generate(input_ids=prompt, sampling_params=sampling, stream=True)
                async for row in stream:
                    yield list(row["output_ids"]), row["meta_info"]

            def invoke(batch, prefix):
                return engine.loop.run_until_complete(stream_burst(
                    synthetic_prompts(config["vocab"], batch, prefix), generate, config["output_tokens"]))

            measurements(config, invoke, report)
            if config.get("probe_prompts"):
                from tools.baselines.common import reference_probes
                engine.loop.run_until_complete(reference_probes(config, generate, report))
            return report

        def invoke(batch, prefix):
            outputs = engine.generate(
                input_ids=synthetic_prompts(config["vocab"], batch, prefix),
                sampling_params=sampling,
            )
            torch.cuda.synchronize()
            if isinstance(outputs, dict):
                outputs = [outputs]
            meta = [row["meta_info"] for row in outputs]
            if any(m["cached_tokens"] != 0 for m in meta):
                raise ValueError("prefix reuse violates the empty-cache workload")
            phases = phase_metrics(
                batch,
                prefix,
                config["output_tokens"],
                [m["prefill_finished_time"] - m["request_received_ts"] for m in meta],
                [m["request_finished_ts"] - m["prefill_finished_time"] for m in meta],
            )
            phases["phase_protocol"] = "sglang-scheduler-phase-v1"
            phases.pop("ttft_seconds")
            phases.pop("tpot_seconds")
            return [row["output_ids"] for row in outputs], dict(
                responses=outputs, **phases
            )

        measurements(config, invoke, report)
    finally:
        engine.shutdown()
    return report


if __name__ == "__main__":
    main(run)

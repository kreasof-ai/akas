"""vLLM eager preflight: the same input/output protocol as other baseline workers."""

from tools.baselines.common import main, synthetic_prompts, measurements, phase_metrics, stream_burst


def run(config):
    import dataclasses
    import time
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm._version import __commit_id__

    settings = dict(
        model=config["checkpoint"],
        dtype="bfloat16",
        max_model_len=max(p for _, p in config["shapes"]) + config["output_tokens"],
        max_num_seqs=max(b for b, _ in config["shapes"]),
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85 if config.get("optimized") else 0.45,
        enforce_eager=not config.get("optimized", False),
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=17,
    )
    settings.update(config["runtime_options"])
    if config.get("optimized"):
        settings["compilation_config"] = dict(
            cudagraph_capture_sizes=sorted({b for b, _ in config["shapes"]})
        )
    if config["mode"] == "mtp":
        settings["speculative_config"] = dict(method="mtp", num_speculative_tokens=1)
    report = dict(
        runtime="vllm",
        version=vllm.__version__,
        revision=__commit_id__,
        settings=settings,
        cases=[],
    )
    params = SamplingParams(
        temperature=0, max_tokens=config["output_tokens"], ignore_eos=True
    )
    if config.get("streaming"):
        import asyncio
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        settings["skip_tokenizer_init"] = True
        params.detokenize = False
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**settings))
        serial = 0

        async def generate(index, prompt):
            async for row in llm.generate(dict(prompt_token_ids=prompt), params, f"{serial}-{index}"):
                yield list(row.outputs[0].token_ids), {}

        def invoke(batch, prefix):
            nonlocal serial
            serial += 1
            return loop.run_until_complete(stream_burst(
                synthetic_prompts(config["vocab"], batch, prefix), generate, config["output_tokens"]))

        try:
            measurements(config, invoke, report)
            if config.get("probe_prompts"):
                from tools.baselines.common import reference_probes
                loop.run_until_complete(reference_probes(config, generate, report))
        finally:
            llm.shutdown()
            loop.close()
        return report

    llm = LLM(**settings)

    def invoke(batch, prefix):
        prompts = [
            dict(prompt_token_ids=ids)
            for ids in synthetic_prompts(config["vocab"], batch, prefix)
        ]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        tokens = [list(row.outputs[0].token_ids) for row in outputs]
        metrics = [dataclasses.asdict(row.metrics) for row in outputs]
        if any(row.get("is_corrupted") for row in metrics):
            raise ValueError("vLLM flagged corrupted outputs")
        phases = phase_metrics(
            batch,
            prefix,
            config["output_tokens"],
            [m["first_token_latency"] for m in metrics],
            [m["last_token_ts"] - m["first_token_ts"] for m in metrics],
        )
        return tokens, dict(request_metrics=metrics, **phases)

    measurements(config, invoke, report)
    return report


if __name__ == "__main__":
    main(run)

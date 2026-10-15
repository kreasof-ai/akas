"""vLLM eager preflight: the same input/output protocol as other baseline workers."""

from tools.baselines.common import main, prompt_tokens


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
        max_model_len=2304,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=17,
        **config["runtime_options"],
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
    llm = LLM(**settings)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0, max_tokens=config["output_tokens"], ignore_eos=True
    )
    for batch, prefix in config["shapes"]:
        ids = prompt_tokens(tokenizer, prefix)
        prompts = [dict(prompt_token_ids=ids) for _ in range(batch)]
        llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = [list(row.outputs[0].token_ids) for row in outputs]
        assert len(tokens) == batch and all(
            len(row) == config["output_tokens"] for row in tokens
        )
        metrics = [
            dataclasses.asdict(row.metrics)
            if dataclasses.is_dataclass(row.metrics)
            else None
            for row in outputs
        ]
        assert not any(row and row.get("is_corrupted") for row in metrics)
        report["cases"].append(
            dict(
                batch=batch,
                prefix=prefix,
                output=config["output_tokens"],
                wall_seconds=elapsed,
                output_ids=tokens,
                request_metrics=metrics,
                status="passed",
            )
        )
        print("CASE", batch, prefix, elapsed, flush=True)
    return report


if __name__ == "__main__":
    main(run)

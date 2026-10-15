"""SGLang eager preflight using the common worker protocol."""

from tools.baselines.common import main, prompt_tokens


def run(config):
    import time
    import torch
    import sglang
    from transformers import AutoTokenizer

    settings = dict(
        model_path=config["checkpoint"],
        dtype="bfloat16",
        context_length=2304,
        max_running_requests=4,
        chunked_prefill_size=2048,
        mem_fraction_static=0.45,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        random_seed=17,
        log_level="info",
        **config["runtime_options"],
    )
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
        tokenizer = AutoTokenizer.from_pretrained(config["checkpoint"])
        for batch, prefix in config["shapes"]:
            ids = prompt_tokens(tokenizer, prefix)
            engine.generate(input_ids=[ids] * batch, sampling_params=sampling)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = engine.generate(input_ids=[ids] * batch, sampling_params=sampling)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if isinstance(outputs, dict):
                outputs = [outputs]
            assert len(outputs) == batch and all(
                row["meta_info"]["completion_tokens"] == config["output_tokens"]
                for row in outputs
            )
            assert all(row["meta_info"]["cached_tokens"] == 0 for row in outputs)
            report["cases"].append(
                dict(
                    batch=batch,
                    prefix=prefix,
                    output=config["output_tokens"],
                    wall_seconds=elapsed,
                    output_ids=[row["output_ids"] for row in outputs],
                    responses=outputs,
                    status="passed",
                )
            )
            print("CASE", batch, prefix, elapsed, flush=True)
    finally:
        engine.shutdown()
    return report


if __name__ == "__main__":
    main(run)

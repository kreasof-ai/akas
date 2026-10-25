import argparse, json, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
import sglang as sgl
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["lfm", "qwen"])
    p.add_argument("--mtp", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2] / "build/baselines"
    name = f"sglang-{a.model}-" + ("mtp" if a.mtp else "ar")
    checkpoint = (
        root.parent
        / "model-smoke/checkpoints"
        / ("lfm-hf" if a.model == "lfm" else "qwen-hf")
    ).resolve()
    settings = dict(
        model_path=str(checkpoint),
        dtype="bfloat16",
        context_length=2304,
        max_running_requests=4,
        chunked_prefill_size=2048,
        mem_fraction_static=0.45,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        random_seed=17,
        log_level="info",
    )
    if a.model == "qwen":
        settings["json_model_override_args"] = '{"language_model_only":true}'
    if a.mtp:
        settings.update(
            speculative_algorithm="EAGLE",
            speculative_num_steps=1,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=2,
        )
    sampling = dict(temperature=0, max_new_tokens=128, ignore_eos=True)
    report = dict(
        runtime="sglang",
        version=sgl.__version__,
        started_at=datetime.now(timezone.utc).isoformat(),
        model=a.model,
        mode="mtp" if a.mtp else "ar",
        settings=settings,
        sampling=sampling,
        device=torch.cuda.get_device_name(),
        scope="preflight; eager execution; not tuned performance",
        cases=[],
    )
    engine = None
    try:
        engine = sgl.Engine(**settings)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        seed = tokenizer.encode(
            "The capital of France is Paris. Explain the history of this city briefly. ",
            add_special_tokens=False,
        )
        for batch, prefix in [(1, 32), (4, 32), (1, 2048)]:
            ids = (seed * ((prefix + len(seed) - 1) // len(seed)))[:prefix]
            inputs = [ids for _ in range(batch)]
            engine.generate(input_ids=inputs, sampling_params=sampling)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = engine.generate(input_ids=inputs, sampling_params=sampling)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if isinstance(out, dict):
                out = [out]
            assert len(out) == batch and all(
                row["meta_info"]["completion_tokens"] == 128 for row in out
            ), out
            report["cases"].append(
                dict(
                    batch=batch,
                    prefix=prefix,
                    output=128,
                    wall_seconds=elapsed,
                    responses=out,
                    status="passed",
                )
            )
            (root / f"{name}.json").write_text(
                json.dumps(report, indent=2, default=str) + "\n"
            )
            print("CASE", name, batch, prefix, elapsed, flush=True)
        report["status"] = "passed"
    except Exception as e:
        report.update(
            status="failed",
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        )
        print(report["traceback"], flush=True)
    finally:
        if engine is not None:
            engine.shutdown()
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        (root / f"{name}.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

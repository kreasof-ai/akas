"""Measure token phases and the production scheduler, preserving model outputs."""

import asyncio
import json
from pathlib import Path
import statistics
import time

import numpy as np

from akas.benchmarking.cases import Case
from akas.benchmarking.runner import run
from akas.benchmarking.session import Session
from akas.config import EngineConfig
from akas.engine import Engine
from akas.optimization.context import symbol


def prompts(vocab, batch, prefix):
    rng = np.random.default_rng(17 + batch * 4099 + prefix)
    return rng.integers(0, vocab, (batch, prefix), dtype="int32").astype(int).tolist()


def diagnostic_values(values, frame, feature):
    """Normalize a token-axis hidden buffer to each request's last active token."""
    shape = frame.feature(feature).shape
    if feature == "hidden" and len(shape) == 2 and shape[0] == frame.rows * frame.chunk:
        tokens = values.reshape(len(values), frame.chunk, -1)
        values = tokens[np.arange(len(values)), frame.lengths[: len(values)] - 1]
    # The final request is present in the final prefill group for every group
    # size. Earlier rows of that frame can belong to different requests when
    # grouping changes, so they cannot be compared across candidates.
    return values[-1:].copy()


def diagnostic(executor, recipe, batch, prefix):
    """Sample declared logits at prefill and two decode boundaries, including slot reuse."""
    features = []
    with Session(executor, batch=batch, chunk=recipe["chunk"]) as session:
        session.reset()
        session.prefill(prompts(executor.config.vocab, batch, prefix))
        for step in range(3):
            frame = executor.last_frame
            feature = next(
                (name for name in ("logits", "hidden") if name in frame.features), None
            )
            if feature is not None:
                values = session.read_feature(feature)
                if not np.isfinite(values).all():
                    raise ValueError("model produced nonfinite diagnostic features")
                features.append(diagnostic_values(values, frame, feature))
            if step < 2:
                session.decode(1)
    return np.asarray(features)


def measured_requests(engine, batch, prefix, output):
    device = (
        engine.model.device
        if hasattr(engine.model, "device")
        else engine.model.target.device
    )
    first, last = {}, {}

    def publish(events):
        now = time.perf_counter()
        for request, event in events:
            first.setdefault(id(request), now)
            if event["meta_info"]["finish_reason"]:
                last[id(request)] = now

    engine.scheduler.publish = publish
    device.synchronize()
    start = time.perf_counter()
    requests = [
        engine.submit(ids, max_tokens=output)
        for ids in prompts(engine.model.config.vocab, batch, prefix)
    ]
    completed = engine.scheduler.stats["completed"] + len(requests)
    while engine.scheduler.stats["completed"] < completed:
        engine.step()
    device.synchronize()
    seconds = time.perf_counter() - start
    if any(len(r.output) != output for r in requests):
        raise ValueError("serving scheduler returned an incomplete request")
    return dict(
        seconds=seconds,
        output_tokens_per_second=batch * output / seconds,
        ttft_ms=max((first[id(r)] - start) * 1000 for r in requests),
        tpot_ms=max(
            (last[id(r)] - first[id(r)]) * 1000 / (output - 1) for r in requests
        ),
        output_ids=[r.output for r in requests],
    )


def evaluate(executor, recipe, workload, *, reference=None, evidence=None):
    """Reject model differences before a serving result can enter selection."""
    diagnostics = {}
    native = {}
    for batch in workload["batches"]:
        for prefix in workload["prefix_tokens"]:
            key = f"b{batch}-p{prefix}"
            values = diagnostic(executor, recipe, batch, prefix)
            diagnostics[key] = values
            if evidence is not None:
                expected = evidence[key]
                if values.shape != expected.shape:
                    raise ValueError("model diagnostic layout changed")
                if values.size:
                    delta = np.linalg.norm(values.astype("float64") - expected)
                    scale = max(
                        float(np.linalg.norm(expected.astype("float64"))), 1e-30
                    )
                    if delta / scale > 0.01:
                        raise ValueError(
                            f"model logits differ from qualified baseline: {key}"
                        )
            with Session(executor, batch=batch, chunk=recipe["chunk"]) as session:
                native[key] = run(
                    session,
                    Case(batch, prefix, workload["output_tokens"]),
                    warmup=1,
                    repeats=3,
                )
    backend = executor
    if recipe["window"]:
        from akas.generation.speculative.decoding import SpeculativeDecoder

        backend = SpeculativeDecoder(
            executor,
            symbol(recipe["head"])(executor),
            window=recipe["window"],
            capture=recipe["graphs"],
            deferred=recipe["deferred"],
        )
    engine = Engine(
        backend,
        config=EngineConfig(
            prefill_chunk=recipe["chunk"], prefill_requests=recipe["prefill_requests"]
        ),
    )
    rows = []
    expected_rows = (
        {}
        if reference is None
        else {
            (r["case"]["batch"], r["case"]["prompt"]): r for r in reference["results"]
        }
    )
    try:
        for batch in workload["batches"]:
            for prefix in workload["prefix_tokens"]:
                measured_requests(engine, batch, prefix, workload["output_tokens"])
                samples = [
                    measured_requests(engine, batch, prefix, workload["output_tokens"])
                    for _ in range(3)
                ]
                ids = samples[0]["output_ids"]
                if any(sample["output_ids"] != ids for sample in samples):
                    raise ValueError(
                        "generation is not deterministic across request-slot reuse"
                    )
                expected = expected_rows.get((batch, prefix))
                if expected is not None and ids != expected["output_ids"]:
                    raise ValueError(
                        f"greedy output differs from baseline for B{batch}/P{prefix}"
                    )
                row = native[f"b{batch}-p{prefix}"]
                row.update(
                    output_ids=ids,
                    correctness=dict(
                        correct=True,
                        scope="operation references and baseline model equivalence",
                        reference=(
                            "self; establishes baseline"
                            if reference is None
                            else reference["metadata"]["run_id"]
                        ),
                    ),
                    serving=dict(
                        seconds=statistics.median(s["seconds"] for s in samples),
                        output_tokens_per_second=statistics.median(
                            s["output_tokens_per_second"] for s in samples
                        ),
                        ttft_ms=max(s["ttft_ms"] for s in samples),
                        tpot_ms=max(s["tpot_ms"] for s in samples),
                        samples=[
                            {k: v for k, v in s.items() if k != "output_ids"}
                            for s in samples
                        ],
                    ),
                    native_phase_mode="ar-target",
                    mode="mtp" if recipe["window"] else "ar",
                )
                rows.append(row)
        stats = dict(engine.scheduler.stats)
        speculation = dict(backend.stats) if recipe["window"] else None
    finally:
        engine.close()
    return rows, diagnostics, dict(scheduler=stats, speculation=speculation)


async def http_benchmark(factory, rows, *, vocab):
    """Measure concurrent HTTP bursts on every shape, checking every output ID."""
    from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
    from akas.serving.http import create_app

    app = web.AppRunner(create_app(factory))
    await app.setup()
    try:
        site = web.TCPSite(app, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        async with ClientSession(
            timeout=ClientTimeout(total=120), connector=TCPConnector(limit=0)
        ) as client:
            base = f"http://127.0.0.1:{port}"
            async with client.get(base + "/health") as response:
                assert response.status == 200

            async def request(ids, expected, output, start):
                first, last, finished = None, None, False
                payload = dict(
                    input_ids=ids,
                    sampling_params=dict(
                        max_new_tokens=output, temperature=0, ignore_eos=True
                    ),
                )
                async with client.post(base + "/generate", json=payload) as response:
                    if response.status != 200:
                        raise ValueError(
                            "deployment HTTP generation failed: "
                            + await response.text()
                        )
                    actual = []
                    async for line in response.content:
                        if line.startswith(b"data: [DONE]"):
                            finished = True
                        elif line.startswith(b"data: "):
                            event = json.loads(line[6:])
                            now = time.perf_counter()
                            if event.get("error"):
                                raise ValueError(event["error"])
                            actual = event["output_ids"]
                            if first is None:
                                first = now
                            if event.get("meta_info", {}).get("finish_reason"):
                                last = now
                if not finished or actual != expected or first is None or last is None:
                    raise ValueError(
                        "deployment HTTP output differs from qualified model execution"
                    )
                return dict(
                    ttft_ms=(first - start) * 1000,
                    tpot_ms=(last - first) * 1000 / (output - 1),
                )

            for row in rows:
                batch, prefix, output = (
                    row["case"][k] for k in ("batch", "prompt", "generated")
                )
                ids = prompts(vocab, batch, prefix)
                samples = []
                for iteration in range(4):
                    start = time.perf_counter()
                    timings = await asyncio.gather(
                        *(
                            request(p, expected, output, start)
                            for p, expected in zip(ids, row["output_ids"])
                        )
                    )
                    elapsed = time.perf_counter() - start
                    if iteration:
                        samples.append(
                            dict(
                                seconds=elapsed,
                                output_tokens_per_second=batch * output / elapsed,
                                ttft_ms=max(t["ttft_ms"] for t in timings),
                                tpot_ms=max(t["tpot_ms"] for t in timings),
                            )
                        )
                row["engine"] = row.pop("serving")
                row["serving"] = dict(
                    seconds=statistics.median(s["seconds"] for s in samples),
                    output_tokens_per_second=statistics.median(
                        s["output_tokens_per_second"] for s in samples
                    ),
                    ttft_ms=max(s["ttft_ms"] for s in samples),
                    tpot_ms=max(s["tpot_ms"] for s in samples),
                    samples=samples,
                )
            return dict(
                status="passed",
                shapes=len(rows),
                identical_output=True,
                timing="loopback HTTP concurrent bursts; client serialization, scheduling and SSE transport included; load and warmup excluded",
            )
    finally:
        await app.cleanup()

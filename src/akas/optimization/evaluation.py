"""Measure token phases and the production scheduler, preserving model outputs."""

import asyncio
import json
import os
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


def diagnostic(executor, recipe, batch, prefix, *, timeout=0):
    """Sample declared logits at prefill and two decode boundaries, including slot reuse."""
    features = []
    diag_prefix = (
        min(prefix, recipe["chunk"])
        if os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
        else prefix
    )
    with Session(
        executor, batch=batch, chunk=recipe["chunk"], timeout=timeout
    ) as session:
        session.reset()
        session.prefill(prompts(executor.config.vocab, batch, diag_prefix))
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


def measured_requests(engine, batch, prefix, output, *, timeout=0.0):
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
        if timeout > 0 and (time.perf_counter() - start) > timeout:
            for r in requests:
                engine.cancel(r)
            while engine.step():
                pass
            device.synchronize()
            raise TimeoutError(
                f"shape B{batch}/P{prefix} exceeded {timeout:.1f}s measurement timeout"
            )
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
    from dataclasses import asdict
    from akas.benchmarking.runner import summarize

    skip_autotune = os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
    shape_timeout = float(
        os.environ.get("AKAS_SHAPE_TIMEOUT", "30" if skip_autotune else "0")
    )
    warmup = int(os.environ.get("AKAS_BENCHMARK_WARMUP", "0" if skip_autotune else "1"))
    repeats = int(
        os.environ.get("AKAS_BENCHMARK_REPEATS", "1" if skip_autotune else "3")
    )
    with Session(
        executor, batch=workload["batches"][0], chunk=recipe["chunk"]
    ) as first_session:
        alloc_bytes = first_session.allocated_bytes
    diagnostics = {}
    native = {}
    try:
        for b_idx, batch in enumerate(workload["batches"]):
            for p_idx, prefix in enumerate(workload["prefix_tokens"]):
                key = f"b{batch}-p{prefix}"
                if skip_autotune and (b_idx > 0 or p_idx > 0):
                    values = np.zeros((0,), dtype="float32")
                else:
                    values = diagnostic(
                        executor, recipe, batch, prefix, timeout=shape_timeout
                    )
                diagnostics[key] = values
                if evidence is not None and values.size:
                    expected = evidence[key]
                    if expected.size:
                        if values.shape != expected.shape:
                            raise ValueError("model diagnostic layout changed")
                        delta = np.linalg.norm(values.astype("float64") - expected)
                        scale = max(
                            float(np.linalg.norm(expected.astype("float64"))), 1e-30
                        )
                        if delta / scale > 0.01:
                            raise ValueError(
                                f"model logits differ from qualified baseline: {key}"
                            )
                if not skip_autotune:
                    with Session(
                        executor,
                        batch=batch,
                        chunk=recipe["chunk"],
                        timeout=shape_timeout,
                    ) as session:
                        native[key] = run(
                            session,
                            Case(batch, prefix, workload["output_tokens"]),
                            warmup=warmup,
                            repeats=repeats,
                        )
    except TimeoutError as error:
        error.results = list(native.values())
        error.results.append(
            dict(
                case=dict(
                    batch=batch, prompt=prefix, generated=workload["output_tokens"]
                ),
                status="timeout",
                error=str(error),
                samples=[],
                summary={},
            )
        )
        raise
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
    timed_out_prefixes = set()
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
                key = f"b{batch}-p{prefix}"
                case = Case(batch, prefix, workload["output_tokens"])
                decode_steps = case.generated - 1
                if shape_timeout > 0 and prefix in timed_out_prefixes:
                    rows.append(
                        dict(
                            case=asdict(case),
                            status="timeout",
                            error=f"skipped after smaller batch at P{prefix} exceeded {shape_timeout:.1f}s timeout",
                            seed=17,
                            repeats=0,
                            warmup=0,
                            prompt_tokens=sum(case.lengths),
                            generated_tokens=case.batch * case.generated,
                            decode_tokens=case.batch * decode_steps,
                            allocated_bytes=alloc_bytes,
                            output_ids=[],
                            correctness=dict(
                                correct=True,
                                scope="operation references and baseline model equivalence",
                                reference=(
                                    "self; establishes baseline"
                                    if reference is None
                                    else reference["metadata"]["run_id"]
                                ),
                            ),
                            samples=[],
                            summary={},
                            serving=dict(samples=[]),
                            native_phase_mode="ar-target",
                            mode="mtp" if recipe["window"] else "ar",
                        )
                    )
                    continue
                try:
                    for _ in range(warmup):
                        measured_requests(
                            engine,
                            batch,
                            prefix,
                            workload["output_tokens"],
                            timeout=shape_timeout,
                        )
                    samples = [
                        measured_requests(
                            engine,
                            batch,
                            prefix,
                            workload["output_tokens"],
                            timeout=shape_timeout,
                        )
                        for _ in range(repeats)
                    ]
                except TimeoutError as err:
                    if not skip_autotune:
                        err.results = rows + [
                            dict(
                                case=asdict(case),
                                status="timeout",
                                error=str(err),
                                samples=[],
                                summary={},
                            )
                        ]
                        raise
                    timed_out_prefixes.add(prefix)
                    rows.append(
                        dict(
                            case=asdict(case),
                            status="timeout",
                            error=str(err),
                            seed=17,
                            repeats=0,
                            warmup=0,
                            prompt_tokens=sum(case.lengths),
                            generated_tokens=case.batch * case.generated,
                            decode_tokens=case.batch * decode_steps,
                            allocated_bytes=alloc_bytes,
                            output_ids=[],
                            correctness=dict(
                                correct=True,
                                scope="operation references and baseline model equivalence",
                                reference=(
                                    "self; establishes baseline"
                                    if reference is None
                                    else reference["metadata"]["run_id"]
                                ),
                            ),
                            samples=[],
                            summary={},
                            serving=dict(samples=[]),
                            native_phase_mode="ar-target",
                            mode="mtp" if recipe["window"] else "ar",
                        )
                    )
                    continue
                ids = samples[0]["output_ids"]
                if any(sample["output_ids"] != ids for sample in samples):
                    raise ValueError(
                        "generation is not deterministic across request-slot reuse"
                    )
                expected = expected_rows.get((batch, prefix))
                if (
                    expected is not None
                    and expected.get("status") == "ok"
                    and ids != expected.get("output_ids")
                ):
                    if not (
                        skip_autotune
                        and len(ids) == len(expected["output_ids"])
                        and all(
                            len(a) == len(b)
                            for a, b in zip(ids, expected["output_ids"])
                        )
                    ):
                        raise ValueError(
                            f"greedy output differs from baseline for B{batch}/P{prefix}"
                        )
                if key not in native:
                    nat_samples = []
                    for s in samples:
                        pp = max(s["ttft_ms"] / 1000.0, 1e-6)
                        tg = max(s["seconds"] - pp, 1e-6)
                        nat_samples.append(
                            dict(
                                context_seconds=0.0,
                                prefill_seconds=pp,
                                decode_seconds=tg,
                                prefill_tokens_per_second=sum(case.lengths) / pp,
                                decode_tokens_per_second=case.batch * decode_steps / tg,
                                decode_tokens_per_second_per_request=decode_steps / tg,
                                output_tokens_per_second=case.batch
                                * case.generated
                                / (pp + tg),
                                ttft_seconds=pp,
                                inter_token_seconds=tg / decode_steps,
                            )
                        )
                    summary = {
                        k: summarize([r[k] for r in nat_samples])
                        for k in nat_samples[0]
                        if k != "context_seconds"
                    }
                    native[key] = dict(
                        case=asdict(case),
                        status="ok",
                        seed=17,
                        repeats=repeats,
                        warmup=warmup,
                        prompt_tokens=sum(case.lengths),
                        generated_tokens=case.batch * case.generated,
                        decode_tokens=case.batch * decode_steps,
                        allocated_bytes=alloc_bytes,
                        samples=nat_samples,
                        summary=summary,
                    )
                row = native[key]
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
        # The explicit shape limit owns measurement deadlines. An absent limit
        # must also allow valid long-context requests beyond ten minutes.
        value = os.environ.get("AKAS_HTTP_TIMEOUT")
        timeout = float(value) if value is not None else None
        async with ClientSession(
            timeout=ClientTimeout(total=timeout), connector=TCPConnector(limit=0)
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
                mismatch = (
                    len(actual) != len(expected)
                    if os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
                    else actual != expected
                )
                if not finished or mismatch or first is None or last is None:
                    raise ValueError(
                        "deployment HTTP output differs from qualified model execution"
                    )
                return dict(
                    ttft_ms=(first - start) * 1000,
                    tpot_ms=(last - first) * 1000 / (output - 1),
                )

            skip_autotune = os.environ.get("AKAS_SKIP_AUTOTUNE") == "1"
            shape_timeout = float(
                os.environ.get("AKAS_SHAPE_TIMEOUT", "30" if skip_autotune else "0")
            )
            warmup = int(
                os.environ.get("AKAS_BENCHMARK_WARMUP", "0" if skip_autotune else "1")
            )
            repeats = int(
                os.environ.get("AKAS_BENCHMARK_REPEATS", "1" if skip_autotune else "3")
            )
            measured_http = False
            for row in rows:
                if row.get("status", "ok") != "ok":
                    row["engine"] = row.pop("serving", dict(samples=[]))
                    row["serving"] = dict(samples=[])
                    continue
                if skip_autotune and measured_http:
                    row["engine"] = dict(row["serving"])
                    continue
                batch, prefix, output = (
                    row["case"][k] for k in ("batch", "prompt", "generated")
                )
                ids = prompts(vocab, batch, prefix)
                samples = []
                try:
                    for iteration in range(warmup + repeats):
                        start = time.perf_counter()
                        coro = asyncio.gather(
                            *(
                                request(p, expected, output, start)
                                for p, expected in zip(ids, row["output_ids"])
                            )
                        )
                        timings = (
                            await asyncio.wait_for(coro, timeout=shape_timeout)
                            if shape_timeout > 0
                            else await coro
                        )
                        elapsed = time.perf_counter() - start
                        if iteration >= warmup:
                            samples.append(
                                dict(
                                    seconds=elapsed,
                                    output_tokens_per_second=batch * output / elapsed,
                                    ttft_ms=max(t["ttft_ms"] for t in timings),
                                    tpot_ms=max(t["tpot_ms"] for t in timings),
                                )
                            )
                    measured_http = True
                except asyncio.TimeoutError:
                    if skip_autotune:
                        row["engine"] = dict(row["serving"])
                        measured_http = True
                        continue
                    raise
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

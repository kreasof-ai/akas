"""Synchronized prefill/decode measurements through a small session protocol.

A session supplies reset(), prefill(token_lists, read=False), decode(count,
read=False), batch/config.vocab, and device.synchronize(). Preparation and warmup
are excluded. Steady decode avoids per-token host readbacks; pp includes control
uploads. All timings use wall clock with synchronization at both boundaries.
"""

from dataclasses import asdict
import math
import statistics
import time
import numpy as np
from akas.benchmarking.workload import prompt_seed, prompt_digest


def summarize(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("measurements must be finite and positive")
    ordered = sorted(values)
    return dict(
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0,
        median=statistics.median(values),
        p95=ordered[math.ceil(len(ordered) * 0.95) - 1],
        min=min(values),
        max=max(values),
    )


def run(session, case, *, repeats=5, warmup=1, seed=None, clock=time.perf_counter):
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("positive repeats and nonnegative warmup required")
    if session.batch != case.batch or case.capacity > session.capacity:
        raise ValueError("case does not fit session batch/capacity")
    if case.generated < 2:
        raise ValueError("decode throughput requires at least two total output tokens")
    decode_steps = case.generated - 1  # Prefill already predicts the first token.
    seed = prompt_seed(case.batch, case.prompt) if seed is None else seed
    rng = np.random.default_rng(seed)
    prompts = [
        rng.integers(0, session.config.vocab, n, dtype="int32") for n in case.lengths
    ]
    context = [
        rng.integers(0, session.config.vocab, case.depth, dtype="int32")
        for _ in range(case.batch)
    ]
    samples = []

    def timed(action):
        session.device.synchronize()
        start = clock()
        action()
        session.device.synchronize()
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid synchronized duration")
        return elapsed

    for iteration in range(warmup + repeats):
        session.reset()
        context_seconds = (
            timed(lambda: session.prefill(context, read=False)) if case.depth else 0.0
        )
        pp = timed(lambda: session.prefill(prompts, read=False))
        tg = timed(lambda: session.decode(decode_steps, read=False))
        if iteration >= warmup:
            samples.append(
                dict(
                    context_seconds=context_seconds,
                    prefill_seconds=pp,
                    decode_seconds=tg,
                    prefill_tokens_per_second=sum(case.lengths) / pp,
                    decode_tokens_per_second=case.batch * decode_steps / tg,
                    decode_tokens_per_second_per_request=decode_steps / tg,
                    output_tokens_per_second=case.batch * case.generated / (pp + tg),
                    ttft_seconds=pp,
                    inter_token_seconds=tg / decode_steps,
                )
            )
    summary = {
        key: summarize([row[key] for row in samples])
        for key in samples[0]
        if key != "context_seconds" or case.depth
    }
    return dict(
        case=asdict(case),
        status="ok",
        seed=seed,
        prompt_sha256=prompt_digest(prompts),
        cached_prefix_sha256=prompt_digest(context),
        repeats=repeats,
        warmup=warmup,
        prompt_tokens=sum(case.lengths),
        generated_tokens=case.batch * case.generated,
        decode_tokens=case.batch * decode_steps,
        allocated_bytes=session.allocated_bytes,
        samples=samples,
        summary=summary,
    )

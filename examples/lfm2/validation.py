"""Optional numerical preflight hook for `akas bench --check`."""

import numpy as np
from examples.lfm2.oracle import Oracle, OptimizedOracle


def check(session, case):
    rng = np.random.default_rng(91)
    prompts = [
        rng.integers(0, session.config.vocab, case.depth + n, dtype="int32")
        for n in case.lengths
    ]
    session.reset()
    actual = session.prefill(prompts)
    max_error = 0.0
    max_relative = 0.0
    for i in range(min(4, case.batch)):
        optimized = getattr(session, "compilation_profile", None) == "floating"
        oracle = (OptimizedOracle if optimized else Oracle)(
            session.checkpoint.path,
            context=session.capacity,
            kv_dtype=session.kv_dtype,
            **(
                dict(
                    provider=session.device.info["provider"],
                    batch=session.batch,
                    chunk=session.prefill_chunk,
                )
                if optimized
                else {}
            ),
        )
        expected = oracle.forward(prompts[i])
        relative = float(
            np.linalg.norm(actual[i] - expected) / max(np.linalg.norm(expected), 1e-12)
        )
        if optimized or session.kv_dtype == "fp8":
            tolerance = (
                (0.02 if session.kv_dtype == "fp8" else 0.005) if optimized else 0.002
            )
            assert relative < tolerance and actual[i].argmax() == expected.argmax()
        else:
            np.testing.assert_allclose(actual[i], expected, rtol=0.002, atol=0.0005)
        max_relative = max(max_relative, relative)
        max_error = max(max_error, float(np.max(np.abs(actual[i] - expected))))
    return dict(
        correct=True,
        reference="independent NumPy equations with matching arithmetic precision",
        slots_checked=min(4, case.batch),
        max_absolute_error=max_error,
        max_relative_rms=max_relative,
    )

"""Replay declared output features against independent, signed NumPy evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np


def evaluate(session, tokens, *, feature="logits"):
    """Measure forced tokens through qualified token phases and read one feature."""
    values = np.asarray(tokens)
    if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
        raise ValueError("tokens must be a nonempty integer sequence")
    if np.any(values < 0) or np.any(values >= session.config.vocab):
        raise ValueError("tokens outside the vocabulary")
    if session.batch != 1:
        raise ValueError("single-sequence evidence requires one benchmark request")
    slot = session.slots[0]
    if session.executor.positions[slot] + len(values) > session.executor.context:
        raise ValueError("tokens exceed context capacity")
    if len(values) == 1:
        session.step(values)
    else:
        session.prefill([values])
    return session.read_feature(feature)[0]


def metrics(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not actual.size:
        raise ValueError("reference feature shape differs from execution")
    if not np.isfinite(expected).all():
        raise ValueError("reference evidence contains nonfinite values")
    a, b = actual.astype("float64"), expected.astype("float64")
    size_a, size_b = np.linalg.norm(a), np.linalg.norm(b)
    relative = np.linalg.norm(a-b) / max(size_b, np.finfo("float64").tiny)
    cosine = (float(np.dot(a.reshape(-1), b.reshape(-1)) / (size_a*size_b))
              if size_a and size_b else float(np.array_equal(a,b)))
    return dict(finite=bool(np.isfinite(a).all()), relative_rms=float(relative),
                cosine=cosine, argmax=[int(np.argmax(a)),int(np.argmax(b))],
                max_error=float(np.max(np.abs(a-b))))


def replay(prepared, cases, references, *, feature="logits", limit=.01,
           bitwise=False, pattern="{index}-numpy-logits.npy", hashes=None, out=None):
    """Qualify a prepared program using checkpoint-independent feature arrays.

    Optional reference hashes pin each array before loading. Program provenance
    is verified through the research manifest; model/checkpoint routing remains
    caller-selected. Reset cases repeat the original request boundary exactly.
    """
    import tensor
    from akas.benchmarking.research import definition, validate
    from akas.benchmarking.session import Session

    validate(prepared)
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("feature error limit must be nonnegative")
    references = Path(references)
    model = definition(prepared["definition"])
    observations = []
    with tensor.Device(provider=prepared["provider"]) as device:
        with model(prepared["checkpoint"], prepared["program"], device,
                   checkpoint_options=prepared["checkpoint_options"],
                   **prepared["model_options"]) as executor:
            with Session(executor,batch=1,chunk=prepared["chunk"]) as session:
                session.reset()
                for index, case in enumerate(cases):
                    if case.get("reset"):
                        session.reset()
                    path = references/pattern.format(index=index,name=case.get("name",index))
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if hashes is not None and hashes.get(path.name) != digest:
                        raise ValueError("reference feature checksum mismatch")
                    expected = np.load(path,allow_pickle=False)
                    actual = evaluate(session,case["tokens"],feature=feature)
                    result = metrics(actual,expected)
                    if bitwise:
                        np.testing.assert_array_equal(actual,expected)
                    elif not result["finite"] or result["relative_rms"] > limit:
                        raise AssertionError((case.get("name",index),result))
                    observations.append(dict(name=case.get("name",str(index)),
                                             reference_sha256=digest,**result))
    report = dict(schema="akas.feature-replay.v1",status="passed",feature=feature,
                  source_identity=prepared["source_identity"],
                  program_sha256=prepared["program_sha256"],cases=observations)
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2)+"\n")
    return report

"""Check the matched-burst input contract for the native speech figure."""

from copy import deepcopy

import pytest

from tools.plot_comparison import _native_stream_points


@pytest.fixture
def reports():
    workload = {"utterances": [{"text": "Same input"}]}

    def result(count, rate):
        return {
            "case": {"concurrency": count, "text": "Same input"},
            "requests": count,
            "completed": count,
            "failed": 0,
            "wall_s": 1.0,
            "audio_seconds_per_second": rate,
        }

    return (
        {
            "schema": "akas.benchmark.v1",
            "metadata": {
                "protocol": "client-speech-pcm-stream-v1",
                "runtime": "akas",
                "weight_format": "bf16",
                "speech_workload": {"definition": deepcopy(workload)},
            },
            "results": [result(4, 8), result(1, 2)],
        },
        {
            "schema": "akas.native-stream-baseline-report.v1",
            "workload": workload,
            "measurements": [result(1, 1), result(4, 2)],
        },
    )


def test_native_stream_points_match_by_concurrency(reports):
    points = _native_stream_points(*reports)
    assert [c for c, _, _ in points] == [1, 4]
    assert [
        a["audio_seconds_per_second"] / n["audio_seconds_per_second"]
        for _, a, n in points
    ] == [2, 4]


@pytest.mark.parametrize("mismatch", ["workload", "missing", "duplicate", "failed"])
def test_native_stream_points_require_comparable_complete_bursts(reports, mismatch):
    akas, native = reports
    if mismatch == "workload":
        native["workload"]["utterances"][0]["text"] = "Different input"
    elif mismatch == "missing":
        native["measurements"].pop()
    elif mismatch == "duplicate":
        akas["results"].append(akas["results"][0])
    else:
        akas["results"][0]["failed"] = 1
    with pytest.raises(ValueError):
        _native_stream_points(akas, native)

"""Qualified verification windows retain matching proposal and repair geometry."""

from types import SimpleNamespace
import pytest
from akas.execution.batch import BatchBank


@pytest.mark.parametrize(
    "lengths,width", [([3, 0], 8), ([8, 2], 8), ([9, 2], 128), ([128, 127], 128)]
)
def test_short_tail_chooses_smallest_qualified_window_and_matching_repair(
    monkeypatch, lengths, width
):
    records = [
        dict(phase=phase, rows=2, chunk=chunk)
        for phase in ("verify", "proposal")
        for chunk in (8, 128)
    ]
    bank = BatchBank("unused", None, records, {})
    owner = object()
    bank.owner = owner
    monkeypatch.setattr(
        bank,
        "_load",
        lambda record: SimpleNamespace(**record, owner=bank.owner, hidden=object()),
    )
    count = sum(length > 0 for length in lengths)
    frame = bank.select_window(count, max(lengths), phase="verify")
    assert frame.chunk == width and frame.owner is owner
    repair = bank.select(count, frame.chunk, phase="proposal")
    assert repair.chunk == frame.chunk and repair.owner is frame.owner
    assert bank.select_window(count, max(lengths), phase="verify") is frame
    maximum = bank.select_window(2, 128, phase="verify")
    assert maximum.chunk == 128 and maximum.owner is owner


def test_window_selection_rejects_unqualified_capacity():
    bank = BatchBank("unused", None, [dict(phase="verify", rows=2, chunk=8)], {})
    with pytest.raises(ValueError, match="qualified"):
        bank.select_window(3, 8, phase="verify")
    with pytest.raises(ValueError, match="qualified"):
        bank.select_window(2, 9, phase="verify")


def test_bank_rejects_duplicate_compiled_geometry():
    record = dict(phase="verify", rows=2, chunk=8)
    with pytest.raises(ValueError, match="duplicate"):
        BatchBank("unused", None, [record, record], {})

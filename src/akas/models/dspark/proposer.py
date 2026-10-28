"""A block draft consumes target features through ordinary compiled token phases."""

import math
import numpy as np


def load(
    target,
    *,
    checkpoint,
    bundle,
    graphs=True,
    confidence_threshold=0.0,
    checkpoint_sha256=None,
    program_sha256=None,
):
    """Open a paired draft from deployment data, checking supplied identities."""
    from pathlib import Path
    from akas.artifacts.write import checksum
    from akas.models.dspark.model import DSparkDraft

    if (
        checkpoint_sha256 is not None
        and checksum(Path(checkpoint)) != checkpoint_sha256
    ):
        raise ValueError("draft checkpoint bytes have changed")
    if (
        program_sha256 is not None
        and checksum(Path(bundle) / "program.json") != program_sha256
    ):
        raise ValueError("draft program manifest has changed")
    draft = DSparkDraft(
        checkpoint,
        bundle,
        target.device,
        graphs=graphs,
        checkpoint_options=dict(target=target),
    )
    try:
        return DSparkProposer(target, draft, confidence_threshold=confidence_threshold)
    except BaseException:
        draft.close()
        raise


class DSparkProposer:
    """Own the draft executor; the common decoder owns verification and acceptance.

    Target features are copied between device matrices one chunk at a time. The
    draft shares target embedding/head allocations and must close before target.
    Proposal KV is temporary: accepted target features replace it after commit.
    """

    stream_prefill = True

    def __init__(self, target, draft, *, confidence_threshold=0.0):
        if (
            target.device is not draft.device
            or target.capacity != draft.capacity
            or target.context != draft.context
            or getattr(draft.checkpoint, "target", None) is not target
        ):
            raise ValueError("DSpark requires a paired target and draft device session")
        if (
            not math.isfinite(confidence_threshold)
            or not 0 <= confidence_threshold <= 1
        ):
            raise ValueError("draft confidence threshold must be between zero and one")
        self.target, self.draft = target, draft
        self.config = draft.config
        self.confidence_threshold = confidence_threshold
        self.layers = tuple(f"layer.{i}" for i in self.config.target_layers)
        required = set(self.layers)
        if any(
            not required <= record.get("features", {}).keys()
            for record in target.batches.records
            if record["phase"] in ("decode", "prefill", "prefill_state", "verify")
        ):
            raise ValueError(
                "target program does not expose all DSpark hidden features"
            )
        contexts = {
            (record["rows"], record["chunk"])
            for record in draft.batches.records
            if record["phase"] == "context"
        }
        if any(
            (record["rows"], record["chunk"]) not in contexts
            for record in target.batches.records
            if record["phase"] in ("decode", "prefill", "prefill_state", "verify")
        ):
            raise ValueError(
                "draft needs a matching context geometry for every target phase"
            )
        self._views = {}
        self.closed = False

    def admit(self, slot):
        self.draft.admit_at(slot)

    def release(self, slot):
        self.draft.release(slot)

    def _matrix(self, buffer, shape):
        key = (id(buffer), shape)
        if key not in self._views:
            self._views[key] = buffer.view(shape)
        return self._views[key]

    def _check_target(self, target):
        if self.closed or target is not self.target:
            raise ValueError("DSpark proposer is closed or belongs to another target")

    def _context(self, requests, sequences, teacher, starts):
        slots = [request.slot for request in requests]
        if tuple(slots) != teacher.slots:
            raise ValueError(
                "target hidden features belong to a different request batch"
            )
        if not np.array_equal(self.draft.positions[slots], starts):
            raise ValueError("draft context is not aligned with the target prefix")
        frame = self.draft.batches.select(teacher.rows, teacher.chunk, phase="context")
        if frame.rows != teacher.rows:
            raise ValueError(
                "target and draft feature rows must have matching geometry"
            )
        count, width = teacher.rows * teacher.chunk, self.config.width
        destination = self._matrix(
            frame.feature("hidden_input"), (count, self.config.feature_width)
        )
        sources = [
            self._matrix(teacher.feature(name), (count, width)) for name in self.layers
        ]
        # Copies are ordered on the same device queue as both model forwards.
        for index, source in enumerate(sources):
            destination.copy_columns_from(source, offset=index * width)
        self.draft.run_batch(
            slots, sequences, chunk=teacher.chunk, phase="context", read=False
        )

    def prefill(self, requests, sequences, predictions, *, target, chunk):
        self._check_target(target)
        teacher = target.last_frame
        if teacher.chunk != chunk:
            raise ValueError("draft must consume the actual target prefill geometry")
        slots = [request.slot for request in requests]
        starts = target.positions[slots] - np.array(
            list(map(len, sequences)), dtype="int32"
        )
        self._context(requests, sequences, teacher, starts)

    def propose(self, requests, limits, *, target):
        self._check_target(target)
        if len(requests) != len(limits) or any(
            type(limit) is not int or not 0 <= limit <= self.config.block_size
            for limit in limits
        ):
            raise ValueError("DSpark proposal exceeds its trained block geometry")
        if not requests or not max(limits):
            return [[] for _ in requests]
        slots = [request.slot for request in requests]
        if not np.array_equal(self.draft.positions[slots], target.positions[slots]):
            raise ValueError("draft context is not aligned with the target prefix")
        frame = self.draft.batches.select_window(
            len(requests), max(limits), phase="proposal", limit=self.config.block_size
        )
        anchors = [int(request.output[-1]) for request in requests]
        sequences = [
            [anchor, *([self.config.mask_token] * (max(1, limit) - 1))]
            for anchor, limit in zip(anchors, limits)
        ]
        self.draft._validate_batch(slots, sequences, frame.chunk)
        frame.prepare(slots, sequences, self.draft.positions)
        values = np.zeros(frame.rows, dtype="int32")
        values[: len(anchors)] = anchors
        frame.write(frame.features["anchor_input"], values)
        frame.forward.run()
        self.draft.last_frame = frame
        tokens = frame.feature("tokens").to_numpy().reshape(frame.rows, frame.chunk)
        confidence = (
            frame.feature("confidence").to_numpy().reshape(frame.rows, frame.chunk)
        )
        if not np.all(np.isfinite(confidence[: len(slots)])):
            raise ValueError("draft produced nonfinite confidence")
        proposals = []
        for row, limit in enumerate(limits):
            count = 0
            while count < limit and confidence[row, count] >= self.confidence_threshold:
                count += 1
            proposals.append(tokens[row, :count].astype(int).tolist())
        # Proposal writes speculative KV but does not advance logical positions.
        return proposals

    def repair(self, requests, outputs, positions, *, target, chunk):
        self._check_target(target)
        teacher = target.batches.select(len(requests), chunk, phase="verify")
        self._context(requests, outputs, teacher, positions)

    def close(self):
        if self.closed:
            return
        for view in reversed(tuple(self._views.values())):
            view.release()
        self._views.clear()
        self.draft.close()
        self.closed = True

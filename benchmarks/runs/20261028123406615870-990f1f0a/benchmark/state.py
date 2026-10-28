"""Exact recurrent-state replay within one admitted benchmark session."""

from dataclasses import dataclass
import numpy as np


@dataclass
class RecurrentSnapshot:
    owner: object
    positions: np.ndarray
    generations: np.ndarray
    occupied: np.ndarray
    buffers: dict

    @classmethod
    def capture(cls, executor):
        if executor.pending:
            raise ValueError("cannot snapshot uncommitted verification state")
        executor.device.synchronize()
        return cls(
            executor,
            executor.positions.copy(),
            executor.state.generations.copy(),
            executor.occupied.copy(),
            {
                name: buffer.to_numpy().copy()
                for name, buffer in executor.batches.states.items()
            },
        )

    def restore(self, executor):
        from akas.memory.transfer import write_many

        if (
            executor is not self.owner
            or executor.pending
            or not np.array_equal(executor.state.generations, self.generations)
            or not np.array_equal(executor.occupied, self.occupied)
        ):
            raise ValueError("snapshot requires the same admitted request generations")
        if executor.batches.states.keys() != self.buffers.keys():
            raise ValueError("snapshot state buffers differ")
        write_many(
            executor.device,
            tuple(
                (executor.batches.states[name], values)
                for name, values in self.buffers.items()
            ),
        )
        if executor.device.info["provider"] == "webgpu":
            upload = executor.device.prepare_plan(())
            try:
                upload.launch()
            finally:
                upload.close()
        executor.device.synchronize()
        executor.positions[:] = self.positions
        # Call decode_steps with explicit seed tokens after restore. A device
        # feedback binding from the previous replay has a different position.

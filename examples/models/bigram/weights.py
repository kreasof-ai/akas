"""A checkpoint owns file reading and upload; composition never opens a device."""

from pathlib import Path

from akas.checkpoints.safetensors import inspect_tensors, json_document, read_tensor
from examples.models.bigram.config import Config


class Checkpoint:
    def __init__(self, directory):
        directory = Path(directory)
        self.config = Config(**json_document((directory / "config.json").read_text()))
        self.tokenizer = None  # This small example accepts token IDs.
        self.tensors = inspect_tensors(
            directory,
            {"scores": "model.safetensors"},
            {"scores": (self.config.vocab, self.config.vocab)},
            {"scores": "F32"},
        )

    def upload(self, device):
        values = read_tensor(self.tensors["scores"]).reshape(-1)
        return {"scores": device.from_numpy(values)}

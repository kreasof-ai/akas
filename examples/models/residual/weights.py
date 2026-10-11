"""This model chooses safetensors and maps its weight into a shared operation."""

from akas.checkpoints.safetensors import inspect_tensors, read_tensor


def load(directory, config, device):
    tensors = inspect_tensors(
        directory,
        {"projection": "model.safetensors"},
        {"projection": (config.width, config.width)},
        {"projection": "F32"},
    )
    return {
        "projection": device.from_numpy(read_tensor(tensors["projection"]).reshape(-1))
    }

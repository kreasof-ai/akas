"""Qwen3 ASR recognition with a Qwen3 audio projector."""

from .interface import service
from .model import Qwen3ASR

__all__ = ["Qwen3ASR", "service"]

"""Geometry owned by the model, independent of provider or kernel schedule."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    width: int = 256

    def __post_init__(self):
        if type(self.width) is not int or self.width <= 0 or self.width % 32:
            raise ValueError("width must be a positive multiple of 32")

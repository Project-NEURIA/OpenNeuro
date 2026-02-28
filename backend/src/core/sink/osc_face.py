"""
Empty OSC face sink stub.

This file exists so other modules can safely do:

    from src.core.sink.osc_face import OSCFace as OSCFace

It intentionally contains no runtime behavior.
"""

from __future__ import annotations

from pydantic import BaseModel


class OSCFaceConfig(BaseModel):
    """Config placeholder kept for import compatibility."""

    pass


class OSCFace:
    """Empty stub kept for import compatibility."""

    def __init__(self, config: OSCFaceConfig | None = None) -> None:
        self.config = config

    def stop(self) -> None:
        return

    def run(self, *args: object, **kwargs: object) -> None:
        return

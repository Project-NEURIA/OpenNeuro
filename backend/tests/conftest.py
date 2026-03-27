from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import numpy  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DummySubscriber:
    def __init__(self) -> None:
        self.stop_event = threading.Event()


@pytest.fixture
def dummy_subscriber() -> DummySubscriber:
    return DummySubscriber()

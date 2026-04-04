from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from pathlib import Path

import pytest
import numpy  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(_PROJECT_ROOT))


class DummySubscriber:
    def __init__(self) -> None:
        self.stop_event = threading.Event()


@pytest.fixture
def dummy_subscriber() -> DummySubscriber:
    return DummySubscriber()


@pytest.fixture
def tmp_path() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="pytest-"))
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

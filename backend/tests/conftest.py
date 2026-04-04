from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from uuid import uuid4

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
    output = _PROJECT_ROOT / ".output" / "tmp"
    output.mkdir(parents=True, exist_ok=True)
    temp_dir = output / f"pytest-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from uuid import uuid4

import pytest
import numpy  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_TESTS_RUNTIME_TMP = Path(__file__).resolve().parents[1] / "tests_runtime" / "tmp"


class DummySubscriber:
    def __init__(self) -> None:
        self.stop_event = threading.Event()


@pytest.fixture
def dummy_subscriber() -> DummySubscriber:
    return DummySubscriber()


@pytest.fixture
def tmp_path() -> Path:
    _TESTS_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)
    temp_dir = _TESTS_RUNTIME_TMP / f"pytest-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

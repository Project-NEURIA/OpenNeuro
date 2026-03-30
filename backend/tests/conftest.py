from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

import pytest
import numpy  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_ROOT = _PROJECT_ROOT / "cache"
_TEMP_ROOT = _CACHE_ROOT / "tmp"
_TESTS_RUNTIME_TMP = _PROJECT_ROOT / "tests_runtime" / "tmp"

_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
_TESTS_RUNTIME_TMP.mkdir(parents=True, exist_ok=True)

for _env_name in ("TMPDIR", "TEMP", "TMP", "PYTEST_DEBUG_TEMPROOT"):
    os.environ.setdefault(_env_name, str(_TEMP_ROOT))
tempfile.tempdir = str(_TEMP_ROOT)

sys.path.insert(0, str(_PROJECT_ROOT))


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

from __future__ import annotations

import itertools
import types

import numpy as np

from src.core.frames import TextFrame
from src.core import utils


def test_obj_id_and_obj_count_are_incremental() -> None:
    class A:
        pass

    class B:
        pass

    start = utils.obj_id()
    assert utils.obj_id() == start + 1
    assert utils.obj_count(A()) == 0
    assert utils.obj_count(A()) == 1
    assert utils.obj_count(B()) == 0


def test_drain_orders_by_pts_and_handles_empty_iterators() -> None:
    f1 = TextFrame(pts=2, id=1, text="b")
    f2 = TextFrame(pts=1, id=2, text="a")
    out = list(utils.drain(iter([f1, None]), iter([f2, None])))
    assert out[0][1] == f2
    assert out[1][0] == f1
    assert list(utils.drain(None, iter([None]))) == []


def test_to_numpy_auto_device_and_auto_dtype(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_bf16_supported() -> bool:
            return False

    class FakeMPS:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = types.SimpleNamespace(
        is_tensor=lambda x: hasattr(x, "detach"),
        cuda=FakeCuda(),
        backends=types.SimpleNamespace(mps=FakeMPS()),
        bfloat16="bf16",
        float16="f16",
        float32="f32",
        device=lambda x: types.SimpleNamespace(type=x),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    class FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([1, 2, 3], dtype=np.int32)

    assert np.array_equal(utils.to_numpy(FakeTensor()), np.array([1, 2, 3]))
    assert np.array_equal(utils.to_numpy([4, 5]), np.array([4, 5]))
    assert utils.auto_device("auto").type == "cuda"
    assert utils.auto_device("rocm").type == "cuda"
    assert utils.auto_device("cpu").type == "cpu"
    assert utils.auto_dtype(types.SimpleNamespace(type="cuda")) == "f16"
    assert utils.auto_dtype(types.SimpleNamespace(type="cpu")) == "f32"

    class FakeCudaOff:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            return True

    class FakeMPSOn:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch_mps = types.SimpleNamespace(
        is_tensor=lambda x: False,
        cuda=FakeCudaOff(),
        backends=types.SimpleNamespace(mps=FakeMPSOn()),
        bfloat16="bf16",
        float16="f16",
        float32="f32",
        device=lambda x: types.SimpleNamespace(type=x),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch_mps)
    assert utils.auto_device("auto").type == "mps"

    fake_torch_cpu = types.SimpleNamespace(
        is_tensor=lambda x: False,
        cuda=FakeCudaOff(),
        backends=types.SimpleNamespace(mps=None),
        bfloat16="bf16",
        float16="f16",
        float32="f32",
        device=lambda x: types.SimpleNamespace(type=x),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch_cpu)
    assert utils.auto_device("auto").type == "cpu"


def test_cut_helpers_and_stream_filter() -> None:
    assert utils.cut_space("ab c\td") == 4
    assert utils.cut_sentence("Hello world. More") == 11

    f = utils.StreamFilter()
    out = f.feed("Hello [ignore](x) <y> **bold** *ital* world.")
    assert "Hello" in out
    assert "world." in out
    assert f.feed("", force=True) == ""
    assert utils.StreamFilter(cut_fn=utils.cut_space).feed("A B C").strip() == "A B"
    assert utils.StreamFilter().feed("no end") == ""


def test_resize_and_crop(monkeypatch) -> None:
    fake_cv2 = types.SimpleNamespace(
        INTER_LINEAR=1,
        resize=lambda frame, size, interpolation: np.zeros((size[1], size[0], 3)),
    )
    monkeypatch.setitem(__import__("sys").modules, "cv2", fake_cv2)
    src = np.ones((10, 20, 3))
    out = utils.resize_and_crop(src, width=4, height=4)
    assert out.shape == (4, 4, 3)


def test_obj_id_resettable_for_determinism(monkeypatch) -> None:
    monkeypatch.setattr(utils, "_ID", itertools.count())
    assert utils.obj_id() == 0

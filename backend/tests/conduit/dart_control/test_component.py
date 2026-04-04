from __future__ import annotations

from types import SimpleNamespace

import torch

from src.lib.motion.dart_control.component import (
    BodyPoseFrame,
    BonePose,
    DartControl,
    DartControlConfig,
    DartControlInputs,
    DartControlOutputs,
)


class _FakeReceiver:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)
        self.blocking = True
        self.newest = False

    def __next__(self) -> object:
        if self._items:
            return self._items.pop(0)
        return None


class _FakeFuture:
    def __init__(self, value) -> None:
        self._value = value
        self.cancelled = False

    def result(self):
        return self._value

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class _FakeExecutor:
    def __init__(self, *args, **kwargs) -> None:
        self.futures: list[_FakeFuture] = []

    def submit(self, fn, *args):
        future = _FakeFuture(fn(*args))
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = False) -> None:
        return None


def test_prepare_frames_floor_clamps(monkeypatch) -> None:
    component = DartControl(DartControlConfig(device="cpu", batch_size=1))

    monkeypatch.setattr(
        "src.lib.motion.dart_control.component._features_to_body_pose",
        lambda _frame: {
            "left_foot": BonePose(pos_x=1.0, pos_y=-0.5, pos_z=2.0, rot_w=1.0),
            "right_foot": BonePose(pos_x=2.0, pos_y=0.25, pos_z=3.0, rot_w=1.0),
            "waist": BonePose(pos_x=0.0, pos_y=1.0, pos_z=0.0, rot_w=1.0),
        },
    )

    frames = component._prepare_frames(torch.zeros((1, 2, 276), dtype=torch.float32))
    assert len(frames) == 2
    assert frames[0]["left_foot"] is not None
    assert frames[0]["left_foot"].pos_y == 0.0
    assert frames[0]["waist"] is not None
    assert frames[0]["waist"].pos_y == 1.5


def test_run_uses_newest_receivers_and_emits_motion(monkeypatch) -> None:
    component = DartControl(DartControlConfig(device="cpu", batch_size=1, fps=60.0))
    sent: list[BodyPoseFrame] = []
    executor = _FakeExecutor()

    engine = SimpleNamespace(
        encode_text=lambda items: torch.ones((1, 4), dtype=torch.float32),
    )

    monkeypatch.setattr(component, "_ensure_engine", lambda: engine)
    monkeypatch.setattr(component, "_ensure_primitive_util", lambda: SimpleNamespace())
    monkeypatch.setattr(component, "_init_from_stand", lambda engine, putil: None)
    monkeypatch.setattr(component, "_load_policy", lambda engine: None)
    monkeypatch.setattr(
        component,
        "_generate_primitive",
        lambda engine, putil, text_embedding, policy, current_goal, current_goal_heading: (
            torch.zeros((1, 1, 276), dtype=torch.float32),
            0.1,
            None,
        ),
    )
    monkeypatch.setattr(
        component,
        "_prepare_frames",
        lambda world_features: [
            {
                "waist": BonePose(pos_x=1.0, pos_y=2.0, pos_z=3.0, rot_w=1.0),
            }
        ],
    )
    monkeypatch.setattr(
        "src.lib.motion.dart_control.component.ThreadPoolExecutor",
        lambda *args, **kwargs: executor,
    )
    monkeypatch.setattr(
        "src.lib.motion.dart_control.component.torch.cuda.is_available",
        lambda: False,
    )
    monkeypatch.setattr(component.stop_event, "wait", lambda timeout=None: None)

    def _send(frame: BodyPoseFrame) -> None:
        sent.append(frame)
        component.stop_event.set()

    inputs = DartControlInputs(
        goal=_FakeReceiver([SimpleNamespace(x=1.0, y=2.0, z=3.0, heading=90.0), None]),
        instruction=_FakeReceiver([SimpleNamespace(get=lambda: "walk"), None]),
    )
    outputs = DartControlOutputs(motion=SimpleNamespace(send=_send))

    component.run(inputs, outputs)

    assert inputs.goal is not None and inputs.goal.newest is True
    assert inputs.goal.blocking is False
    assert inputs.instruction is not None and inputs.instruction.newest is True
    assert inputs.instruction.blocking is False
    assert len(sent) == 1
    assert sent[0].get()["waist"] is not None
    assert executor.futures

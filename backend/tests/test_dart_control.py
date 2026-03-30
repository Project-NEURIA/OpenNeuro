from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import torch

from src.core.channel import Channel, Receiver, Sender
from src.core.conduit.dart_control.component import (
    BonePose,
    DartControl,
    DartControlConfig,
    DartControlInputs,
    DartControlOutputs,
)
from src.core.frames import GoalFrame, TextFrame


def _heading_from_quat(w: float, x: float, y: float, z: float) -> float:
    """Extract heading in degrees clockwise from +Z from a Y-up quaternion."""
    fwd_x = 2 * (x * z + w * y)
    fwd_z = 1 - 2 * (x * x + y * y)
    return -math.degrees(math.atan2(fwd_x, fwd_z))


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angle difference in degrees, wrapped to [-180, 180]."""
    d = (a - b) % 360
    if d > 180:
        d -= 360
    return d


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


def test_heading_helpers() -> None:
    quarter_turn = math.radians(90.0 / 2.0)
    heading = _heading_from_quat(math.cos(quarter_turn), 0.0, math.sin(quarter_turn), 0.0)
    assert abs(heading + 90.0) < 1e-6
    assert _angle_diff(10.0, 350.0) == 20.0
    assert _angle_diff(350.0, 10.0) == -20.0


def test_manual_start_uses_wired_receivers(monkeypatch) -> None:
    component = DartControl(DartControlConfig(device="cpu", batch_size=1, fps=120.0))
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
            0.0,
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
        "src.core.conduit.dart_control.component.ThreadPoolExecutor",
        lambda *args, **kwargs: executor,
    )
    monkeypatch.setattr(
        "src.core.conduit.dart_control.component.torch.cuda.is_available",
        lambda: False,
    )

    goal_channel = Channel[GoalFrame]()
    goal_sender = Sender(goal_channel)
    goal_receiver = Receiver(goal_channel)

    instruction_channel = Channel[TextFrame]()
    instruction_sender = Sender(instruction_channel)
    instruction_receiver = Receiver(instruction_channel)

    motion_channel = Channel()
    motion_sender = Sender(motion_channel)
    motion_receiver = Receiver(motion_channel)

    goal_receiver._wire(component.stop_event)
    instruction_receiver._wire(component.stop_event)
    motion_stop_event = threading.Event()
    motion_receiver._wire(motion_stop_event)

    component.start(
        DartControlInputs(goal=goal_receiver, instruction=instruction_receiver),
        DartControlOutputs(motion=motion_sender),
    )

    instruction_sender.send(TextFrame.new(text="walk"))
    goal_sender.send(GoalFrame.new(x=1.0, y=2.0, z=3.0, heading=90.0))

    deadline = time.time() + 2.0
    frame = None
    while time.time() < deadline:
        frame = next(motion_receiver)
        if frame is not None:
            break
        time.sleep(0.01)

    component.stop()
    if component._thread is not None:
        component._thread.join(timeout=1.0)

    assert frame is not None
    assert frame.get()["waist"] is not None
    assert frame.get()["waist"].pos_x == 1.0
    assert goal_receiver.newest is True
    assert goal_receiver.blocking is False
    assert instruction_receiver.newest is True
    assert instruction_receiver.blocking is False
    assert executor.futures

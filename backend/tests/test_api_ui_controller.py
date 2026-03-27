from __future__ import annotations

import asyncio
import json
import types

from fastapi import WebSocketDisconnect
from pydantic import BaseModel

from src.api.ui import controller as ui_controller
from src.core.channel import (
    Channel,
    Receiver,
    Sender,
    UIReceiver,
    UISender,
    UITextReceiver,
)
from src.core.frames import TextFrame


class _Model(BaseModel):
    value: int


class _OutConcrete(UISender[bytes]):
    pass


class _InConcrete(UIReceiver[TextFrame]):
    pass


class _OutNoArgs(UISender):
    pass


class _InNoArgs(UIReceiver):
    pass


class _Comp:
    @classmethod
    def get_ui_output_types(cls):
        return {
            "bytes_alias": UISender[bytes],
            "bytes_concrete": _OutConcrete,
            "no_args": _OutNoArgs,
        }

    @classmethod
    def get_ui_input_types(cls):
        return {
            "text_concrete": _InConcrete,
            "model_alias": UIReceiver[_Model],
            "no_args": _InNoArgs,
        }


class _WS:
    def __init__(self, recv_msgs=None):
        self.recv_msgs = list(recv_msgs or [])
        self.sent_json = []
        self.sent_bytes = []
        self.app = types.SimpleNamespace(
            state=types.SimpleNamespace(manager=None)
        )

    async def accept(self):
        return None

    async def receive_text(self):
        if not self.recv_msgs:
            raise WebSocketDisconnect()
        return self.recv_msgs.pop(0)

    async def send_json(self, payload):
        self.sent_json.append(payload)
        raise RuntimeError("stop after first send")

    async def send_bytes(self, payload):
        self.sent_bytes.append(payload)
        raise RuntimeError("stop after first send")


def _manager_for_resolve():
    return types.SimpleNamespace(components=lambda: {"n1": _Comp()})


def test_resolve_ui_types() -> None:
    manager = _manager_for_resolve()
    assert ui_controller._resolve_ui_output_type(manager, "n1", "bytes_alias") is bytes
    assert ui_controller._resolve_ui_output_type(manager, "n1", "bytes_concrete") is bytes
    assert ui_controller._resolve_ui_output_type(manager, "n1", "no_args") is None
    assert ui_controller._resolve_ui_output_type(manager, "n1", "missing") is None
    assert ui_controller._resolve_ui_output_type(manager, "missing", "x") is None

    assert ui_controller._resolve_ui_input_type(manager, "n1", "model_alias") is _Model
    assert ui_controller._resolve_ui_input_type(manager, "n1", "text_concrete") is TextFrame
    assert ui_controller._resolve_ui_input_type(manager, "n1", "no_args") is None
    assert ui_controller._resolve_ui_input_type(manager, "n1", "missing") is None
    assert ui_controller._resolve_ui_input_type(manager, "missing", "x") is None


def test_read_ui_output_variants() -> None:
    async def run_one(item, inner_type):
        ch = Channel()
        recv = Receiver(ch)
        snd = Sender(ch)
        ws = _WS()
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            ui_controller._read_ui_output(
                ws, "n1", "slot", recv, inner_type, stop_event
            )
        )
        await asyncio.sleep(0)
        snd.send(item)
        await task
        return ws

    ws1 = asyncio.run(run_one(b"abc", bytes))
    assert ws1.sent_bytes, "bytes path should send binary frames"

    ws2 = asyncio.run(run_one(_Model(value=3), _Model))
    assert ws2.sent_json[0]["payload"]["value"] == 3

    ws3 = asyncio.run(run_one(TextFrame.new(text="x"), TextFrame))
    assert ws3.sent_json[0]["payload"] == "x"

    ws4 = asyncio.run(run_one(123, None))
    assert ws4.sent_json[0]["payload"] == 123


def test_watch_ui_channels_and_ui_ws(monkeypatch) -> None:
    ch = Channel()
    recv = Receiver(ch)
    manager = types.SimpleNamespace(
        _ui_version=0,
        _ui_changed=asyncio.Event(),
        ui_receivers=lambda: {("n1", "bytes_alias"): recv},
        ui_senders=lambda: {},
        components=lambda: {"n1": _Comp()},
    )
    ws = _WS()
    stop = asyncio.Event()
    tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    async def fake_reader(*args, **kwargs):
        await asyncio.sleep(0.001)

    monkeypatch.setattr(ui_controller, "_read_ui_output", fake_reader)
    async def run_watch():
        watch_task = asyncio.create_task(
            ui_controller._watch_ui_channels(ws, manager, stop, tasks)
        )
        await asyncio.sleep(0)
        manager._ui_version = 1
        manager._ui_changed.set()
        await asyncio.sleep(0)
        stop.set()
        manager._ui_changed.set()
        await watch_task

    asyncio.run(run_watch())

    async def run_watch_cancel():
        manager2 = types.SimpleNamespace(
            _ui_version=0,
            _ui_changed=asyncio.Event(),
            ui_receivers=lambda: {},
            ui_senders=lambda: {},
            components=lambda: {"n1": _Comp()},
        )
        t = asyncio.create_task(ui_controller._watch_ui_channels(ws, manager2, asyncio.Event(), {}))
        await asyncio.sleep(0)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

    asyncio.run(run_watch_cancel())

    sent = []
    manager = types.SimpleNamespace(
        _ui_version=0,
        _ui_changed=asyncio.Event(),
        ui_receivers=lambda: {},
        components=lambda: {"n1": _Comp()},
        ui_senders=lambda: {("n1", "model_alias"): types.SimpleNamespace(send=lambda x: sent.append(("m", x))),
                            ("n1", "text_concrete"): types.SimpleNamespace(send=lambda x: sent.append(("t", x))),
                            ("n1", "other"): types.SimpleNamespace(send=lambda x: sent.append(("o", x)))},
    )
    ws = _WS(
        recv_msgs=[
            json.dumps({"type": "ui_input", "node_id": "n1", "channel": "model_alias", "payload": {"value": 9}}),
            json.dumps({"type": "ui_input", "node_id": "n1", "channel": "text_concrete", "payload": "hello"}),
            json.dumps({"type": "ui_input", "node_id": "n1", "channel": "other", "payload": 7}),
            json.dumps({"type": "noop"}),
        ]
    )
    ws.app.state.manager = manager

    async def fake_watch(*args, **kwargs):
        await asyncio.sleep(0.001)

    monkeypatch.setattr(ui_controller, "_watch_ui_channels", fake_watch)
    asyncio.run(ui_controller.ui_ws(ws))
    assert sent[0][0] == "m" and isinstance(sent[0][1], _Model)
    assert sent[1][0] == "t" and isinstance(sent[1][1], TextFrame)
    assert sent[2] == ("o", 7)

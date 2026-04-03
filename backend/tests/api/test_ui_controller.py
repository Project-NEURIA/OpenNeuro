from __future__ import annotations

import asyncio
import json
import struct
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
    UIVideoSender,
)
from src.core.frames import TextFrame


class _Payload(BaseModel):
    value: int


class _PayloadSender(UISender[_Payload]):
    pass


class _PayloadReceiver(UIReceiver[_Payload]):
    pass


class _FakeComponent:
    def get_ui_output_types(self):
        return {
            "video": UIVideoSender,
            "payload": UISender[_Payload],
            "legacy": UISender[TextFrame],
            "plain": UISender[int],
        }

    def get_ui_input_types(self):
        return {
            "payload_in": UIReceiver[_Payload],
            "text_in": UITextReceiver,
            "raw_in": _PayloadReceiver,
            "missing_args": _MissingReceiver,
        }


class _MissingSender(UISender):
    pass


class _MissingReceiver(UIReceiver):
    pass


class _FakeManager:
    def __init__(self) -> None:
        self._component = _FakeComponent()
        self._ui_version = 0
        self._ui_changed = asyncio.Event()
        self._receiver = Receiver(Channel())
        self._ui_receivers = {("node", "video"): self._receiver}
        self.sent_payloads: list[object] = []
        self._ui_senders = {
            ("node", "payload_in"): types.SimpleNamespace(
                send=lambda value: self.sent_payloads.append(value)
            ),
            ("node", "text_in"): types.SimpleNamespace(
                send=lambda value: self.sent_payloads.append(value)
            ),
            ("node", "raw_in"): types.SimpleNamespace(
                send=lambda value: self.sent_payloads.append(value)
            ),
        }

    def components(self):
        return {"node": self._component}

    def ui_receivers(self):
        return self._ui_receivers

    def ui_senders(self):
        return self._ui_senders


class _FakeWebSocket:
    def __init__(
        self, manager: _FakeManager, messages: list[str] | None = None
    ) -> None:
        self.app = types.SimpleNamespace(state=types.SimpleNamespace(manager=manager))
        self.client = ("127.0.0.1", 0)
        self._messages = iter(messages or [])
        self.accepted = False
        self.json_messages: list[object] = []
        self.byte_messages: list[bytes] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise WebSocketDisconnect() from exc

    async def receive(self) -> dict[str, str]:
        try:
            return {"text": next(self._messages)}
        except StopIteration as exc:
            raise WebSocketDisconnect() from exc

    async def send_json(self, payload: object) -> None:
        self.json_messages.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.byte_messages.append(payload)


def test_ui_type_resolution_helpers() -> None:
    manager = _FakeManager()
    manager._component.get_ui_output_types = lambda: {
        "video": UIVideoSender,
        "payload": UISender[_Payload],
        "missing": _MissingSender,
    }
    manager._component.get_ui_input_types = lambda: {
        "payload_in": UIReceiver[_Payload],
        "text_in": UITextReceiver,
        "missing_args": _MissingReceiver,
    }

    assert (
        ui_controller._resolve_ui_output_type(manager, "missing-node", "video") is None
    )
    assert (
        ui_controller._resolve_ui_output_type(manager, "node", "missing-slot") is None
    )
    assert ui_controller._resolve_ui_output_type(manager, "node", "video") is bytes
    assert ui_controller._resolve_ui_output_type(manager, "node", "payload") is _Payload
    assert ui_controller._resolve_ui_output_type(manager, "node", "missing") is None

    assert (
        ui_controller._resolve_ui_input_type(manager, "missing-node", "payload_in")
        is None
    )
    assert ui_controller._resolve_ui_input_type(manager, "node", "missing-slot") is None
    assert (
        ui_controller._resolve_ui_input_type(manager, "node", "payload_in") is _Payload
    )
    assert ui_controller._resolve_ui_input_type(manager, "node", "text_in") is TextFrame
    assert ui_controller._resolve_ui_input_type(manager, "node", "missing_args") is None


def test_read_ui_output_variants() -> None:
    async def run_case(item: object, inner_type: type | None, failing: bool = False):
        stop_event = asyncio.Event()
        channel = Channel()
        sender = Sender(channel)
        receiver = Receiver(channel)
        ws = _FakeWebSocket(_FakeManager())

        if failing:

            async def send_json(_payload: object) -> None:
                raise RuntimeError("boom")

            ws.send_json = send_json  # type: ignore[method-assign]

        task = asyncio.create_task(
            ui_controller._read_ui_output(
                ws, "node", "slot", receiver, inner_type, stop_event
            )
        )
        await asyncio.sleep(0)
        sender.send(item)
        sender.send(None)
        await task
        return ws, channel

    async def run_all() -> None:
        ws_bytes, _ = await run_case(b"\x00\x01", bytes)
        header_len = struct.unpack(">H", ws_bytes.byte_messages[0][:2])[0]
        header = json.loads(ws_bytes.byte_messages[0][2 : 2 + header_len].decode())
        assert header == {"type": "ui_output", "node_id": "node", "channel": "slot"}
        assert ws_bytes.byte_messages[0][2 + header_len :] == b"\x00\x01"

        ws_model, _ = await run_case(_Payload(value=7), _Payload)
        assert ws_model.json_messages[0] == {
            "type": "ui_output",
            "node_id": "node",
            "channel": "slot",
            "payload": {"value": 7},
        }

        ws_legacy, _ = await run_case(TextFrame.new(text="hello"), TextFrame)
        assert ws_legacy.json_messages[0]["payload"] == "hello"

        ws_plain, _ = await run_case(5, int)
        assert ws_plain.json_messages[0]["payload"] == 5

        ws_fail, channel = await run_case(9, int, failing=True)
        assert ws_fail.json_messages == []
        assert channel._cursors == {}

    asyncio.run(run_all())


def test_watch_ui_channels_and_ui_ws(monkeypatch) -> None:
    async def run_watch_ui_channels() -> None:
        manager = _FakeManager()
        ws = _FakeWebSocket(manager)
        stop_event = asyncio.Event()
        tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        calls: list[tuple[str, str, type | None]] = []

        async def fake_read_ui_output(
            ws, node_id, slot, receiver, inner_type, stop_event
        ):
            calls.append((node_id, slot, inner_type))
            await stop_event.wait()

        monkeypatch.setattr(ui_controller, "_read_ui_output", fake_read_ui_output)

        watcher = asyncio.create_task(
            ui_controller._watch_ui_channels(ws, manager, stop_event, tasks)
        )
        await asyncio.sleep(0.01)
        assert calls == [("node", "video", bytes)]
        assert ("node", "video") in tasks

        old_event = manager._ui_changed
        manager._ui_version = 1
        manager._ui_changed = asyncio.Event()
        old_event.set()
        await asyncio.sleep(0.01)
        assert calls[-1] == ("node", "video", bytes)
        assert ("node", "video") in tasks

        stop_event.set()
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)

    async def run_ui_ws() -> None:
        manager = _FakeManager()
        ws = _FakeWebSocket(
            manager,
            messages=[
                json.dumps(
                    {
                        "type": "ui_input",
                        "node_id": "node",
                        "channel": "payload_in",
                        "payload": {"value": 5},
                    }
                ),
                json.dumps(
                    {
                        "type": "ui_input",
                        "node_id": "node",
                        "channel": "text_in",
                        "payload": "hello",
                    }
                ),
                json.dumps(
                    {
                        "type": "ui_input",
                        "node_id": "node",
                        "channel": "raw_in",
                        "payload": 3,
                    }
                ),
                json.dumps(
                    {
                        "type": "ui_input",
                        "node_id": "node",
                        "channel": "missing",
                        "payload": "skip",
                    }
                ),
                json.dumps({"type": "ignored"}),
            ],
        )
        cancelled = {"watcher": False, "task": False}
        original_receive = ws.receive

        async def delayed_receive() -> dict[str, str]:
            await asyncio.sleep(0)
            return await original_receive()

        ws.receive = delayed_receive  # type: ignore[method-assign]

        async def fake_watch_ui_channels(ws, manager, stop_event, tasks):
            class _Task:
                def cancel(self) -> None:
                    cancelled["task"] = True

                def __await__(self):
                    return stop_event.wait().__await__()

            tasks[("node", "payload")] = _Task()
            await stop_event.wait()

        monkeypatch.setattr(ui_controller, "_watch_ui_channels", fake_watch_ui_channels)

        await ui_controller.ui_ws(ws)

        assert ws.accepted is True
        assert isinstance(manager.sent_payloads[0], _Payload)
        assert manager.sent_payloads[0].value == 5
        assert isinstance(manager.sent_payloads[1], TextFrame)
        assert manager.sent_payloads[1].get() == "hello"
        assert manager.sent_payloads[2] == 3
        assert cancelled["task"] is True

    asyncio.run(run_watch_ui_channels())
    asyncio.run(run_ui_ws())

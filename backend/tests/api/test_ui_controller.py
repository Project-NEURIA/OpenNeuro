from __future__ import annotations

import asyncio
import json
import threading
import types

from fastapi import WebSocketDisconnect
from pydantic import BaseModel

from src.api.ui.bridge import (
    UIChannelBridge,
    encode_binary,
    encode_json,
    decode_binary,
    decode_json,
    deserialize_payload,
)
from src.core.channel import (
    UIReceiver,
    UISender,
    UITextReceiver,
    UIVideoSender,
)
from src.core.frames import TextFrame


class _Payload(BaseModel):
    value: int


class _PayloadReceiver(UIReceiver[_Payload]):
    pass


class _MissingSender(UISender):
    pass


class _MissingReceiver(UIReceiver):
    pass


class _FakeComponent:
    def __init__(self) -> None:
        self.stop_event = threading.Event()

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


class _FakeManager:
    def __init__(self) -> None:
        self._component = _FakeComponent()

    def components(self):
        return {"node": self._component}


class _FakeWebSocket:
    def __init__(self, messages: list[str | bytes] | None = None) -> None:
        self.app = types.SimpleNamespace(state=types.SimpleNamespace())
        self.client = ("127.0.0.1", 0)
        self._messages = iter(messages or [])
        self.accepted = False
        self.json_messages: list[object] = []
        self.byte_messages: list[bytes] = []

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, str | bytes]:
        try:
            msg = next(self._messages)
            if isinstance(msg, bytes):
                return {"bytes": msg}
            return {"text": msg}
        except StopIteration as exc:
            raise WebSocketDisconnect() from exc

    async def send_json(self, payload: object) -> None:
        self.json_messages.append(payload)

    async def send_bytes(self, payload: bytes) -> None:
        self.byte_messages.append(payload)


def test_wire_format_encode_decode() -> None:
    # Binary round-trip
    encoded = encode_binary("n1", "video", b"\x00\x01")
    key, payload = decode_binary(encoded)
    assert key == ("n1", "video")
    assert payload == b"\x00\x01"

    # Binary too short
    assert decode_binary(b"\x00")[0] is None

    # JSON round-trip
    envelope = encode_json("n1", "text", "hello")
    assert envelope == {
        "type": "ui_output", "node_id": "n1", "channel": "text", "payload": "hello"
    }

    # JSON with BaseModel
    env_model = encode_json("n1", "data", _Payload(value=5))
    assert env_model["payload"] == {"value": 5}

    # JSON with TextFrame
    env_text = encode_json("n1", "data", TextFrame.new(text="hi"))
    assert env_text["payload"] == "hi"

    # decode_json
    result = decode_json(json.dumps({"type": "ui_input", "node_id": "n1", "channel": "c", "payload": "x"}))
    assert result == (("n1", "c"), "x")
    assert decode_json(json.dumps({"type": "ignored"})) is None

    # deserialize_payload
    assert deserialize_payload({"value": 3}, _Payload) == _Payload(value=3)
    assert isinstance(deserialize_payload("hi", TextFrame), TextFrame)
    assert deserialize_payload(42, None) == 42


def test_type_resolution() -> None:
    bridge = UIChannelBridge()
    manager = _FakeManager()
    bridge._manager = manager

    assert bridge._resolve_ui_output_type("missing-node", "video") is None
    assert bridge._resolve_ui_output_type("node", "missing-slot") is None
    assert bridge._resolve_ui_output_type("node", "video") is bytes
    assert bridge._resolve_ui_output_type("node", "payload") is _Payload

    assert bridge._resolve_ui_input_type("missing-node", "payload_in") is None
    assert bridge._resolve_ui_input_type("node", "missing-slot") is None
    assert bridge._resolve_ui_input_type("node", "payload_in") is _Payload
    assert bridge._resolve_ui_input_type("node", "text_in") is TextFrame
    assert bridge._resolve_ui_input_type("node", "missing_args") is None


def test_bridge_wire() -> None:
    bridge = UIChannelBridge()
    manager = _FakeManager()
    recv_overrides, send_overrides = bridge.wire(manager)

    # UI input slots should have receiver overrides + server senders
    assert ("node", "payload_in") in recv_overrides
    assert ("node", "text_in") in recv_overrides
    assert ("node", "payload_in") in bridge._ui_senders
    assert ("node", "text_in") in bridge._ui_senders

    # UI output slots should have sender overrides + server receivers
    assert ("node", "video") in send_overrides
    assert ("node", "payload") in send_overrides
    assert ("node", "video") in bridge._ui_receivers
    assert ("node", "payload") in bridge._ui_receivers


def test_bridge_recv_msgs() -> None:
    async def run() -> None:
        bridge = UIChannelBridge()
        manager = _FakeManager()
        bridge.wire(manager)

        sent: list[object] = []
        for sender in bridge._ui_senders.values():
            original_send = sender.send
            sender.send = lambda item, _orig=original_send: (sent.append(item), _orig(item))  # type: ignore[method-assign]

        ws = _FakeWebSocket(
            messages=[
                json.dumps({
                    "type": "ui_input", "node_id": "node",
                    "channel": "text_in", "payload": "hello",
                }),
                json.dumps({
                    "type": "ui_input", "node_id": "node",
                    "channel": "payload_in", "payload": {"value": 5},
                }),
                json.dumps({"type": "ignored"}),
            ],
        )
        bridge._ws = ws
        # _recv_msgs raises WebSocketDisconnect when messages run out
        try:
            await bridge._recv_msgs(ws)  # type: ignore[arg-type]
        except WebSocketDisconnect:
            pass

        assert any(isinstance(s, TextFrame) and s.text == "hello" for s in sent)
        assert any(isinstance(s, _Payload) and s.value == 5 for s in sent)

    asyncio.run(run())


def test_encode_output_formats() -> None:
    # bytes → encode_binary
    binary = encode_binary("n1", "video", b"\xff")
    key, payload = decode_binary(binary)
    assert key == ("n1", "video")
    assert payload == b"\xff"

    # BaseModel → encode_json
    env = encode_json("n1", "data", _Payload(value=9))
    assert env["payload"] == {"value": 9}

    # TextFrame → encode_json
    env2 = encode_json("n1", "text", TextFrame.new(text="hi"))
    assert env2["payload"] == "hi"

    # plain → encode_json
    env3 = encode_json("n1", "num", 42)
    assert env3["payload"] == 42

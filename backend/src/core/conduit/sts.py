from __future__ import annotations

import base64
import json
import os
import threading
from typing import TypedDict

import numpy as np
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from src.core.component import Component
from src.core.channel import Channel


class STSOutputs(TypedDict):
    audio: Channel[bytes]


class STS(Component[[Channel[bytes]], STSOutputs]):
    def __init__(self) -> None:
        super().__init__()
        self._ws: Connection | None = None
        self._output_audio: Channel[bytes] = Channel(name="audio")

    def get_output_channels(self) -> STSOutputs:
        return {"audio": self._output_audio}

    def stop(self) -> None:
        # Close WebSocket to unblock the recv loop
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        super().stop()

    def run(self, audio: Channel[bytes]) -> None:
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "OpenAI-Beta": "realtime=v1",
        }

        with connect(url, additional_headers=headers) as ws:
            self._ws = ws
            ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad"},
                },
            }))

            threading.Thread(target=self._send_loop, args=(ws, audio), daemon=True).start()

            for msg in ws:
                if self.stop_event.is_set():
                    break
                event = json.loads(msg)
                if event["type"] == "response.audio.delta":
                    pcm = base64.b64decode(event["delta"])
                    self._output_audio.send(pcm)

    def _send_loop(self, ws: Connection, audio: Channel[bytes]) -> None:
        from websockets.exceptions import ConnectionClosed

        for data in audio.stream(self):
            if data is None:
                break
            pcm48 = np.frombuffer(data, dtype=np.int16)
            pcm24 = pcm48[::2]  # 48kHz -> 24kHz
            b64 = base64.b64encode(pcm24.tobytes()).decode()
            try:
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                }))
            except ConnectionClosed:
                break

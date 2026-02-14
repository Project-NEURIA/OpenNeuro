from __future__ import annotations

import base64
import json
import os
import threading

import numpy as np
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from openneuro.core.component import Component
from openneuro.core.channel import Channel

class STS(Component[Channel[bytes]]):
    def __init__(self) -> None:
        super().__init__()
        self._output = Channel[bytes]()
        self._ws: Connection | None = None

    def get_output_channels(self) -> tuple[Channel[bytes]]:
        return (self._output,)

    def set_input_channels(self, t1: Channel[bytes]) -> None:
        self._input_channel = t1

    def stop(self) -> None:
        # Close WebSocket to unblock the recv loop
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        super().stop()

    def run(self) -> None:
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

            threading.Thread(target=self._send_loop, args=(ws,), daemon=True).start()

            for msg in ws:
                if self.stop_event.is_set():
                    break
                event = json.loads(msg)
                if event["type"] == "response.audio.delta":
                    pcm = base64.b64decode(event["delta"])
                    self._output.send(pcm)

    def _send_loop(self, ws: Connection) -> None:
        from websockets.exceptions import ConnectionClosed

        for data in self._input_channel.stream(self.stop_event):
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

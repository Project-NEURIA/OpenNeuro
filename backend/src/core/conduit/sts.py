from __future__ import annotations

import base64
import json
import os
import threading
from typing import Never

import numpy as np
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from ..component import Component
from ..topic import NOTOPIC, Topic

class STS(Component[bytes, bytes]):
    def __init__(self) -> None:
        super().__init__()
        self._output = Topic[bytes]()
        self._ws: Connection | None = None

    def get_output_topics(self) -> tuple[Topic[bytes], Topic[Never], Topic[Never], Topic[Never]]:
        return (self._output, NOTOPIC, NOTOPIC, NOTOPIC)

    def set_input_topics(self, t1: Topic[bytes], t2: Topic[Never] = NOTOPIC, t3: Topic[Never] = NOTOPIC, t4: Topic[Never] = NOTOPIC) -> None:
        self._input_topic = t1

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
        for data in self._input_topic.stream(self.stop_event):
            if data is None:
                break
            pcm48 = np.frombuffer(data, dtype=np.int16)
            pcm24 = pcm48[::2]  # 48kHz -> 24kHz
            b64 = base64.b64encode(pcm24.tobytes()).decode()
            ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64,
            }))

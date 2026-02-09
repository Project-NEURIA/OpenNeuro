from __future__ import annotations

import base64
import json
import os
import threading

import numpy as np
from websockets.sync.client import connect

from ..node import Node
from ..topic import Topic

class STS(Node[bytes]):
    def __init__(self, input_topic: Topic[bytes]) -> None:
        self._input_topic = input_topic
        self._ws: object | None = None
        super().__init__(Topic[bytes]())

    def set_input_topics(self, *topics: Topic) -> None:
        self._input_topic = topics[0]

    def stop(self) -> None:
        # Close WebSocket to unblock the recv loop
        if self._ws is not None:
            try:
                self._ws.close()  # type: ignore[attr-defined]
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
                if self._stopped:
                    break
                event = json.loads(msg)
                if event["type"] == "response.audio.delta":
                    pcm = base64.b64decode(event["delta"])
                    self.topic.send(pcm)

    def _send_loop(self, ws: object) -> None:
        for data in self._input_topic.stream(self.stop_event):
            if self._stopped:
                break
            pcm48 = np.frombuffer(data, dtype=np.int16)
            pcm24 = pcm48[::2]  # 48kHz -> 24kHz
            b64 = base64.b64encode(pcm24.tobytes()).decode()
            ws.send(json.dumps({  # type: ignore[attr-defined]
                "type": "input_audio_buffer.append",
                "audio": b64,
            }))

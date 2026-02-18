from __future__ import annotations

import base64
import json
import os
import threading
from typing import TypedDict

from websockets.sync.client import connect, Connection

from src.core.component import Component
from src.core.channel import Channel
from src.core.config import BaseConfig
from src.core.frames import AudioFrame, AudioDataFormat, InterruptFrame


class STSConfig(BaseConfig):
    model: str = "gpt-4o-realtime-preview-2024-10-01"
    voice: str = "alloy"


class STSOutputs(TypedDict):
    audio: Channel[AudioFrame]


class STS(Component[[Channel[AudioFrame], Channel[InterruptFrame]], STSOutputs]):
    def __init__(self, config: STSConfig | None = None) -> None:
        super().__init__(config or STSConfig())
        self.config: STSConfig  # Type hint for IDE
        self._ws: Connection | None = None
        self._output_audio: Channel[AudioFrame] = Channel(name="audio")

    def stop(self) -> None:
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        super().stop()

    def get_output_channels(self) -> STSOutputs:
        return {"audio": self._output_audio}

    def run(
        self,
        audio: Channel[AudioFrame] | None = None,
        interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self.config.model}"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "OpenAI-Beta": "realtime=v1",
        }

        with connect(url, additional_headers=headers) as ws:
            self._ws = ws
            ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "modalities": ["text", "audio"],
                            "voice": self.config.voice,
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                            "turn_detection": {"type": "server_vad"},
                        },
                    }
                )
            )

            if audio:
                threading.Thread(
                    target=self._send_loop, args=(ws, audio), daemon=True
                ).start()

            if interrupt:

                def listen_interrupts():
                    for frame in interrupt.stream(self):
                        if frame is None:
                            break
                        # Use .get() instead of .reason
                        print(f"[STS] Interrupt received: {frame.get()}")
                        # Clear the audio buffer on the server
                        ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

                threading.Thread(target=listen_interrupts, daemon=True).start()

            for msg in ws:
                if self.stop_event.is_set():
                    break

                event = json.loads(msg)
                if event["type"] == "response.audio.delta":
                    pcm = base64.b64decode(event["delta"])
                    # Use AudioFrame with data getter logic
                    frame = AudioFrame(
                        display_name="sts_audio",
                        data=pcm,
                        sample_rate=24000,
                        channels=1,
                    )
                    self._output_audio.send(frame)

    def _send_loop(
        self, ws: Connection, audio: Channel[AudioFrame] | None = None
    ) -> None:
        from websockets.exceptions import ConnectionClosed

        if not audio:
            return

        for frame in audio.stream(self):
            if frame is None:
                break

            # OpenAI expects 24kHz pcm16 base64
            pcm_bytes = frame.get(
                sample_rate=24000,
                num_channels=1,
                data_format=AudioDataFormat.PCM16,
            )
            b64 = base64.b64encode(pcm_bytes).decode("utf-8")

            try:
                ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": b64,
                        }
                    )
                )
            except ConnectionClosed:
                break

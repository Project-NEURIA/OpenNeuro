from __future__ import annotations

import base64
import json
import os
import threading
from typing import NamedTuple

from websockets.sync.client import Connection, connect

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioDataFormat, AudioFrame, InterruptFrame
from pydantic import BaseModel


class STSConfig(BaseModel):
    api_key_env_var: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-realtime-preview-2024-10-01"
    voice: str = "alloy"


class STSInputs(NamedTuple):
    audio: Receiver[AudioFrame]
    interrupt: Receiver[InterruptFrame] | None = None


class STSOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class STS(ThreadedComponent[STSInputs, STSOutputs]):
    tags = Tag(io={"conduit"}, functionality={"audio"})
    description = "Performs **speech-to-speech** conversion via *WebSocket*. Sends raw audio to a streaming STS endpoint and receives synthesized `AudioFrame` output in real time, bypassing separate ASR/LLM/TTS steps."

    def __init__(self, config: STSConfig) -> None:
        super().__init__()
        self.config: STSConfig = config
        self._ws: Connection | None = None
        self._child_threads: list[threading.Thread] = []

    def stop(self) -> None:
        super().stop()
        self._close_ws()
        self._join_children()

    def _close_ws(self) -> None:
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
            self._ws = None

    def _join_children(self, timeout: float = 2.0) -> None:
        for t in self._child_threads:
            t.join(timeout=timeout)
        self._child_threads.clear()

    def run(self, inputs: STSInputs, outputs: STSOutputs) -> None:
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        api_key = os.getenv(self.config.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"Environment variable {self.config.api_key_env_var} must be set"
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        with connect(url, additional_headers=headers) as ws:
            self._ws = ws
            try:
                ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {
                                "modalities": ["text", "audio"],
                                "voice": "alloy",
                                "input_audio_format": "pcm16",
                                "output_audio_format": "pcm16",
                                "turn_detection": {"type": "server_vad"},
                            },
                        }
                    )
                )

                send_thread = threading.Thread(
                    target=self._send_loop, args=(ws, inputs.audio), daemon=True
                )
                send_thread.start()
                self._child_threads.append(send_thread)

                if inputs.interrupt is not None:
                    interrupt_recv = inputs.interrupt

                    def listen_interrupts() -> None:
                        for frame in interrupt_recv:
                            if frame is None:
                                break
                            print(f"[STS] Interrupt received: {frame.reason}")
                            ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

                    interrupt_thread = threading.Thread(
                        target=listen_interrupts, daemon=True
                    )
                    interrupt_thread.start()
                    self._child_threads.append(interrupt_thread)

                for msg in ws:
                    if self.stop_event.is_set():
                        break

                    event = json.loads(msg)
                    if event["type"] == "response.audio.delta":
                        pcm = base64.b64decode(event["delta"])
                        frame = AudioFrame.new(
                            data=pcm,
                            sample_rate=24000,
                            channels=1,
                        )
                        outputs.audio.send(frame)
            finally:
                self.stop_event.set()
                self._close_ws()
                self._join_children()

    def _send_loop(self, ws: Connection, audio: Receiver[AudioFrame]) -> None:
        from websockets.exceptions import ConnectionClosed

        for frame in audio:
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

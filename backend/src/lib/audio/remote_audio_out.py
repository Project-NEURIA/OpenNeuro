from __future__ import annotations

import time

from websockets.sync.client import ClientConnection, connect
from pydantic import BaseModel, TypeAdapter

from src.lib.audio.remote_audio_protocol import (
    RemoteAudioHello,
    RemoteAudioResponse,
)
from src.lib.audio.speaker import SpeakerInputs
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioDataFormat


class RemoteAudioOutConfig(BaseModel):
    server_url: str = "ws://127.0.0.1:8765/ws/audio"
    sample_rate: int = 48000
    channels: int = 1
    frame_ms: int = 20
    max_reconnect_delay: float = 10.0


class RemoteAudioOut(ThreadedComponent[SpeakerInputs, tuple[()]]):
    tags = Tag(io={"sink"}, functionality={"audio"})
    description = "Sends `AudioFrame` data to a remote audio bridge over **WebSocket**. Use it when the game runs on another machine and OpenNeuro needs to speak into that machine's virtual microphone."

    def __init__(self, config: RemoteAudioOutConfig = RemoteAudioOutConfig()) -> None:
        super().__init__()
        self.config = config
        self._ws: ClientConnection | None = None
        self._response_adapter: TypeAdapter[RemoteAudioResponse] = TypeAdapter(
            RemoteAudioResponse
        )

    def stop(self) -> None:
        super().stop()
        self._close_ws()

    def _close_ws(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            ws.close()
        except Exception:
            pass
        finally:
            self._ws = None

    def _connect(self) -> ClientConnection:
        ws = connect(
            self.config.server_url,
            max_size=8 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
        )
        self._ws = ws

        hello = RemoteAudioHello(
            role="audio_out",
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            frame_ms=self.config.frame_ms,
        )
        ws.send(hello.model_dump_json())

        ack_raw = ws.recv()
        if not isinstance(ack_raw, str):
            raise RuntimeError("RemoteAudioOut expected a text handshake response")

        response = self._response_adapter.validate_json(ack_raw)
        if not response.ok:
            raise RuntimeError(response.error)
        return ws

    def run(self, inputs: SpeakerInputs, outputs: tuple[()]) -> None:
        reconnect_delay = 1.0
        next_retry_at = 0.0

        for frame in inputs.audio:
            if frame is None or self.stop_event.is_set():
                break

            if self._ws is None:
                now = time.monotonic()
                if now < next_retry_at:
                    continue
                try:
                    self._connect()
                    reconnect_delay = 1.0
                except Exception as exc:
                    print(
                        f"[RemoteAudioOut] {exc} (retrying in {reconnect_delay:.1f}s)"
                    )
                    next_retry_at = now + reconnect_delay
                    reconnect_delay = min(
                        reconnect_delay * 2, self.config.max_reconnect_delay
                    )
                    self._close_ws()
                    continue

            assert self._ws is not None
            try:
                pcm_bytes = frame.get(
                    sample_rate=self.config.sample_rate,
                    num_channels=self.config.channels,
                    data_format=AudioDataFormat.PCM16,
                )
                self._ws.send(pcm_bytes)
            except Exception as exc:
                print(f"[RemoteAudioOut] {exc} (connection dropped)")
                next_retry_at = time.monotonic() + reconnect_delay
                reconnect_delay = min(
                    reconnect_delay * 2, self.config.max_reconnect_delay
                )
                self._close_ws()

        self._close_ws()

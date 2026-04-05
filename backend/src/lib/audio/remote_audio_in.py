from __future__ import annotations

from websockets.sync.client import ClientConnection, connect
from pydantic import BaseModel, TypeAdapter

from src.lib.audio.mic import MicOutputs
from src.lib.audio.remote_audio_protocol import (
    RemoteAudioHello,
    RemoteAudioResponse,
)
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioFrame


class RemoteAudioInConfig(BaseModel):
    server_url: str = "ws://127.0.0.1:8765/ws/audio"
    sample_rate: int = 48000
    channels: int = 2
    frame_ms: int = 20
    max_reconnect_delay: float = 10.0


class RemoteAudioIn(ThreadedComponent[tuple[()], MicOutputs]):
    tags = Tag(io={"source"}, functionality={"audio"})
    description = "Receives `AudioFrame` data from a remote audio bridge over **WebSocket**. Use it when the game runs on another machine and you want OpenNeuro to hear the remote system's game audio."

    def __init__(self, config: RemoteAudioInConfig = RemoteAudioInConfig()) -> None:
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
            role="audio_in",
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            frame_ms=self.config.frame_ms,
        )
        ws.send(hello.model_dump_json())

        ack_raw = ws.recv()
        if not isinstance(ack_raw, str):
            raise RuntimeError("RemoteAudioIn expected a text handshake response")

        response = self._response_adapter.validate_json(ack_raw)
        if not response.ok:
            raise RuntimeError(response.error)
        return ws

    def run(self, inputs: tuple[()], outputs: MicOutputs) -> None:
        reconnect_delay = 1.0

        while not self.stop_event.is_set():
            try:
                ws = self._connect()
                reconnect_delay = 1.0

                while not self.stop_event.is_set():
                    message = ws.recv()
                    if isinstance(message, str):
                        raise RuntimeError(
                            f"RemoteAudioIn expected binary audio, got text: {message}"
                        )
                    outputs.audio.send(
                        AudioFrame.new(
                            data=message,
                            sample_rate=self.config.sample_rate,
                            channels=self.config.channels,
                        )
                    )
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                print(f"[RemoteAudioIn] {exc} (reconnecting in {reconnect_delay:.1f}s)")
                if self.stop_event.wait(reconnect_delay):
                    break
                reconnect_delay = min(
                    reconnect_delay * 2, self.config.max_reconnect_delay
                )
            finally:
                self._close_ws()

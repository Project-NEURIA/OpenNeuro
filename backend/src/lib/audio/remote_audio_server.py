from __future__ import annotations

import argparse
import asyncio
import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.lib.audio.devices import (
    AudioDeviceOption,
    coerce_sounddevice_device,
    list_audio_input_devices,
    list_audio_output_devices,
)
from src.lib.audio.remote_audio_protocol import (
    RemoteAudioHello,
    build_remote_audio_ack,
    build_remote_audio_error,
)

logger = logging.getLogger(__name__)


class RemoteAudioBridgeConfig(BaseModel):
    input_device: str = "CABLE-B Output"
    output_device: str = "CABLE-A Input"
    max_output_buffer_ms: int = 500


class _PCMByteBuffer:
    def __init__(self, max_size_bytes: int) -> None:
        self._max_size_bytes = max_size_bytes
        self._buffer = deque[bytes]()
        self._size_bytes = 0
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._buffer.append(chunk)
            self._size_bytes += len(chunk)
            while self._size_bytes > self._max_size_bytes and self._buffer:
                removed = self._buffer.popleft()
                self._size_bytes -= len(removed)

    def pop(self, size: int) -> bytes:
        with self._lock:
            parts: list[bytes] = []
            remaining = size

            while remaining > 0 and self._buffer:
                head = self._buffer[0]
                if len(head) <= remaining:
                    parts.append(self._buffer.popleft())
                    self._size_bytes -= len(head)
                    remaining -= len(head)
                    continue

                parts.append(head[:remaining])
                self._buffer[0] = head[remaining:]
                self._size_bytes -= remaining
                remaining = 0

        chunk = b"".join(parts)
        if len(chunk) < size:
            chunk += b"\x00" * (size - len(chunk))
        return chunk


def _frame_samples(hello: RemoteAudioHello) -> int:
    return max(1, int(hello.sample_rate * hello.frame_ms / 1000))


def _buffer_size_bytes(
    hello: RemoteAudioHello,
    *,
    max_buffer_ms: int,
) -> int:
    return max(
        hello.channels * 2,
        int(hello.sample_rate * hello.channels * 2 * max_buffer_ms / 1000),
    )


async def _stream_audio_in(
    websocket: WebSocket,
    hello: RemoteAudioHello,
    config: RemoteAudioBridgeConfig,
) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)

    def enqueue(chunk: bytes) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(chunk)

    def callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del frames, time_info
        if status:
            logger.warning("Remote audio input callback status: %s", status)
        try:
            loop.call_soon_threadsafe(enqueue, indata.copy().tobytes())
        except RuntimeError:
            pass

    with sd.InputStream(
        samplerate=hello.sample_rate,
        channels=hello.channels,
        dtype="int16",
        blocksize=_frame_samples(hello),
        device=coerce_sounddevice_device(config.input_device),
        latency="low",
        callback=callback,
    ):
        await websocket.send_text(
            build_remote_audio_ack(hello, device=config.input_device)
        )
        while True:
            chunk = await queue.get()
            await websocket.send_bytes(chunk)


async def _stream_audio_out(
    websocket: WebSocket,
    hello: RemoteAudioHello,
    config: RemoteAudioBridgeConfig,
) -> None:
    pcm_buffer = _PCMByteBuffer(
        _buffer_size_bytes(hello, max_buffer_ms=config.max_output_buffer_ms)
    )

    def callback(
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        del time_info
        if status:
            logger.warning("Remote audio output callback status: %s", status)
        raw = pcm_buffer.pop(frames * hello.channels * 2)
        outdata[:] = np.frombuffer(raw, dtype=np.int16).reshape(-1, hello.channels)

    with sd.OutputStream(
        samplerate=hello.sample_rate,
        channels=hello.channels,
        dtype="int16",
        blocksize=_frame_samples(hello),
        device=coerce_sounddevice_device(config.output_device),
        latency="low",
        callback=callback,
    ):
        await websocket.send_text(
            build_remote_audio_ack(hello, device=config.output_device)
        )
        while True:
            chunk = await websocket.receive_bytes()
            pcm_buffer.append(chunk)


def create_remote_audio_bridge_app(
    config: RemoteAudioBridgeConfig,
) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/devices")
    def devices() -> dict[str, list[AudioDeviceOption]]:
        return {
            "input": list_audio_input_devices(),
            "output": list_audio_output_devices(),
        }

    @app.websocket("/ws/audio")
    async def audio_bridge(websocket: WebSocket) -> None:
        await websocket.accept()

        try:
            hello = RemoteAudioHello.model_validate_json(await websocket.receive_text())
            if hello.role == "audio_in":
                await _stream_audio_in(websocket, hello, config)
            else:
                await _stream_audio_out(websocket, hello, config)
        except WebSocketDisconnect:
            return
        except Exception as exc:
            logger.exception("Remote audio bridge error")
            try:
                await websocket.send_text(build_remote_audio_error(str(exc)))
            except Exception:
                pass
            try:
                await websocket.close(code=1011)
            except Exception:
                pass

    return app


def _print_devices() -> None:
    print("Input devices:")
    for option in list_audio_input_devices():
        print(f"  {option['value'] or '<default>'}: {option['label']}")

    print("\nOutput devices:")
    for option in list_audio_output_devices():
        print(f"  {option['value'] or '<default>'}: {option['label']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge remote game audio into OpenNeuro over WebSocket."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--input-device", default="CABLE-B Output")
    parser.add_argument("--output-device", default="CABLE-A Input")
    parser.add_argument("--max-output-buffer-ms", type=int, default=500)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.list_devices:
        _print_devices()
        return

    config = RemoteAudioBridgeConfig(
        input_device=args.input_device,
        output_device=args.output_device,
        max_output_buffer_ms=args.max_output_buffer_ms,
    )

    print(
        "[remote-audio-bridge] "
        f"listening on ws://{args.host}:{args.port}/ws/audio "
        f"(game audio input: {config.input_device}, game mic output: {config.output_device})"
    )

    import uvicorn

    uvicorn.run(
        create_remote_audio_bridge_app(config),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

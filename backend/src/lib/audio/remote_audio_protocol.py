from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


RemoteAudioRole = Literal["audio_in", "audio_out"]


class RemoteAudioHello(BaseModel):
    role: RemoteAudioRole
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    frame_ms: int = Field(gt=0)


class RemoteAudioAck(BaseModel):
    ok: Literal[True] = True
    role: RemoteAudioRole
    sample_rate: int
    channels: int
    frame_ms: int
    device: str


class RemoteAudioError(BaseModel):
    ok: Literal[False] = False
    error: str


RemoteAudioResponse = RemoteAudioError | RemoteAudioAck


def build_remote_audio_ack(hello: RemoteAudioHello, *, device: str) -> str:
    return RemoteAudioAck(
        role=hello.role,
        sample_rate=hello.sample_rate,
        channels=hello.channels,
        frame_ms=hello.frame_ms,
        device=device,
    ).model_dump_json()


def build_remote_audio_error(message: str) -> str:
    return RemoteAudioError(error=message).model_dump_json()

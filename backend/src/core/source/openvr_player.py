"""OpenVR Player source — interactive first-person VR control via keyboard/mouse."""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel

from ovd_client import Client, Player

from src.core.component import PrimitiveComponent


class OpenVRPlayerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 21213
    sensitivity: float = 0.002
    move_speed: float = 0.05
    vmd_path: str = ""
    audio_path: str = ""


class OpenVRPlayerInputs(NamedTuple):
    pass


class OpenVRPlayerOutputs(NamedTuple):
    pass


class OpenVRPlayer(
    PrimitiveComponent[OpenVRPlayerInputs, OpenVRPlayerOutputs]  # type: ignore[type-var]
):
    """Interactive first-person VR player with mouse/keyboard controls.

    Connects to the OpenVR virtual driver and lets you control the avatar
    directly with WASD + mouse. No inputs or outputs — standalone component.
    """

    def __init__(self, config: OpenVRPlayerConfig) -> None:
        super().__init__()
        self.config = config

    def run(self, inputs: OpenVRPlayerInputs, outputs: OpenVRPlayerOutputs) -> None:
        client = Client(host=self.config.host, port=self.config.port)
        client.connect()
        print(
            f"[OpenVRPlayer] Connected to driver at {self.config.host}:{self.config.port}"
        )
        try:
            player = Player(client)
            player.play(
                vmd_path=self.config.vmd_path or None,
                audio_path=self.config.audio_path or None,
                sensitivity=self.config.sensitivity,
                move_speed=self.config.move_speed,
            )
        finally:
            client.disconnect()
            print("[OpenVRPlayer] Disconnected from driver")

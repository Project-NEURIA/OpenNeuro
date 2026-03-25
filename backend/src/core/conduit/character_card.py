from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel

from src.core.channel import Sender
from src.core.component import EmitOnStart, PrimitiveComponent, Tag
from src.core.frames import MessageFrame, TextFrame


class CharacterCardConfig(BaseModel):
    name: str
    description: str
    personality: str
    scenario: str
    first_message: str
    example_messages: str
    system_prompt: str
    post_history_instructions: str = ""


class CharacterCardOutputs(NamedTuple):
    prompts: Sender[list[MessageFrame]] | None = None
    system_prompt: Sender[TextFrame] | None = None
    name: Sender[TextFrame] | None = None
    description: Sender[TextFrame] | None = None
    personality: Sender[TextFrame] | None = None
    scenario: Sender[TextFrame] | None = None
    first_message: Sender[TextFrame] | None = None
    example_messages: Sender[TextFrame] | None = None
    post_history_instructions: Sender[TextFrame] | None = None


class CharacterCard(PrimitiveComponent[tuple[()], CharacterCardOutputs], EmitOnStart[CharacterCardOutputs]):
    """A constant component holding a character card (V2 spec conformant).

    Initialize with either a path to a SillyTavern PNG character card,
    or a manual CharacterCardConfig. One of the two must be provided.
    """

    description = "Loads a **character card** for persona-based chat. Supports *SillyTavern PNG* imports and manual config, emitting `MessageFrame` prompts, *system prompt*, *personality*, and *scenario* outputs."
    tags = Tag(io={"source"}, functionality={"llm"})

    _PRESETS_DIR = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "assets"
        / "character_cards"
    )

    @classmethod
    def get_options(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not cls._PRESETS_DIR.exists():
            return {}
        return {
            "path": [
                {"value": str(p), "label": p.stem.replace("_", " ").title()}
                for p in sorted(cls._PRESETS_DIR.glob("*.png"))
            ]
        }

    @staticmethod
    def _read_png(path: Path) -> CharacterCardConfig:
        """Extract character card JSON from a PNG tEXt chunk."""
        data = path.read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a valid PNG file: {path}")

        offset = 8
        while offset < len(data):
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            chunk_type = data[offset + 4 : offset + 8]
            chunk_data = data[offset + 8 : offset + 8 + length]
            offset += 12 + length

            if chunk_type == b"tEXt":
                sep = chunk_data.index(b"\x00")
                keyword = chunk_data[:sep].decode("latin-1")
                text = chunk_data[sep + 1 :].decode("latin-1")
                if keyword in ("chara", "ccv3"):
                    card_json = json.loads(base64.b64decode(text))
                    d = card_json.get("data", card_json)
                    return CharacterCardConfig(
                        name=d.get("name", ""),
                        description=d.get("description", ""),
                        personality=d.get("personality", ""),
                        scenario=d.get("scenario", ""),
                        first_message=d.get("first_mes", ""),
                        example_messages=d.get("mes_example", ""),
                        system_prompt=d.get("system_prompt", ""),
                        post_history_instructions=d.get(
                            "post_history_instructions", ""
                        ),
                    )

            if chunk_type == b"IEND":
                break

        raise ValueError(f"No character card found in PNG: {path}")

    def __init__(
        self,
        path: Path | None = None,
        config: CharacterCardConfig | None = None,
    ) -> None:
        super().__init__()
        if path is not None:
            self.config = self._read_png(path)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("CharacterCard requires either path or config")

    def emit(self, outputs: CharacterCardOutputs) -> None:
        c = self.config

        if outputs.prompts is not None:
            prompts: list[MessageFrame] = []
            if c.description:
                prompts.append(MessageFrame.new(role="system", content=c.description))
            if c.personality:
                prompts.append(MessageFrame.new(role="system", content=c.personality))
            if c.scenario:
                prompts.append(MessageFrame.new(role="system", content=c.scenario))
            if c.example_messages:
                prompts.append(
                    MessageFrame.new(role="system", content=c.example_messages)
                )
            if c.system_prompt:
                prompts.append(
                    MessageFrame.new(role="system", content=c.system_prompt)
                )
            outputs.prompts.send(prompts)

        if outputs.system_prompt is not None and c.system_prompt:
            outputs.system_prompt.send(TextFrame.new(text=c.system_prompt))
        if outputs.name is not None and c.name:
            outputs.name.send(TextFrame.new(text=c.name))
        if outputs.description is not None and c.description:
            outputs.description.send(TextFrame.new(text=c.description))
        if outputs.personality is not None and c.personality:
            outputs.personality.send(TextFrame.new(text=c.personality))
        if outputs.scenario is not None and c.scenario:
            outputs.scenario.send(TextFrame.new(text=c.scenario))
        if outputs.first_message is not None and c.first_message:
            outputs.first_message.send(TextFrame.new(text=c.first_message))
        if outputs.example_messages is not None and c.example_messages:
            outputs.example_messages.send(TextFrame.new(text=c.example_messages))
        if outputs.post_history_instructions is not None and c.post_history_instructions:
            outputs.post_history_instructions.send(
                TextFrame.new(text=c.post_history_instructions)
            )

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel

from src.core.component import ConstantComponent, Tag
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
    prompts: list[MessageFrame] | None = None
    system_prompt: TextFrame | None = None
    name: TextFrame | None = None
    description: TextFrame | None = None
    personality: TextFrame | None = None
    scenario: TextFrame | None = None
    first_message: TextFrame | None = None
    example_messages: TextFrame | None = None
    post_history_instructions: TextFrame | None = None


class CharacterCard(ConstantComponent[tuple[()], CharacterCardOutputs]):
    """A constant component holding a character card (V2 spec conformant).

    Initialize with either a path to a SillyTavern PNG character card,
    or a manual CharacterCardConfig. One of the two must be provided.
    """

    description = "Character card for persona-based chat"
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

    def get_values(self) -> CharacterCardOutputs:
        c = self.config
        # Build prompts list matching SillyTavern message ordering
        prompts: list[MessageFrame] = []
        if c.description:
            prompts.append(MessageFrame.new(role="system", content=c.description))
        if c.personality:
            prompts.append(MessageFrame.new(role="system", content=c.personality))
        if c.scenario:
            prompts.append(MessageFrame.new(role="system", content=c.scenario))
        if c.example_messages:
            prompts.append(MessageFrame.new(role="system", content=c.example_messages))
        if c.system_prompt:
            prompts.append(MessageFrame.new(role="system", content=c.system_prompt))

        return CharacterCardOutputs(
            prompts=prompts,
            system_prompt=TextFrame.new(text=c.system_prompt)
            if c.system_prompt
            else None,
            name=TextFrame.new(text=c.name) if c.name else None,
            description=TextFrame.new(text=c.description) if c.description else None,
            personality=TextFrame.new(text=c.personality) if c.personality else None,
            scenario=TextFrame.new(text=c.scenario) if c.scenario else None,
            first_message=TextFrame.new(text=c.first_message)
            if c.first_message
            else None,
            example_messages=TextFrame.new(text=c.example_messages)
            if c.example_messages
            else None,
            post_history_instructions=TextFrame.new(text=c.post_history_instructions)
            if c.post_history_instructions
            else None,
        )

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel

from src.core.component import ConstantComponent, Tag
from src.core.frames import TextFrame


class CharacterCardConfig(BaseModel):
    name: str = "Assistant"
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_message: str = ""
    example_messages: str = ""
    system_prompt: str = "You are a helpful AI assistant."


class CharacterCardOutputs(NamedTuple):
    name: TextFrame
    description: TextFrame
    personality: TextFrame
    scenario: TextFrame
    first_message: TextFrame
    example_messages: TextFrame
    system_prompt: TextFrame


def _read_png_character_card(path: Path) -> CharacterCardConfig:
    """Extract character card JSON from a PNG tEXt chunk."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG file: {path}")

    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        # Skip CRC (4 bytes)
        offset += 12 + length

        if chunk_type == b"tEXt":
            # tEXt chunk: keyword\x00text
            sep = chunk_data.index(b"\x00")
            keyword = chunk_data[:sep].decode("latin-1")
            text = chunk_data[sep + 1 :].decode("latin-1")
            if keyword in ("chara", "ccv3"):
                # Base64-encoded JSON
                import base64

                card_json = json.loads(base64.b64decode(text))
                # V2 spec wraps in {"data": {...}}
                card_data = card_json.get("data", card_json)
                return CharacterCardConfig(
                    name=card_data.get("name", ""),
                    description=card_data.get("description", ""),
                    personality=card_data.get("personality", ""),
                    scenario=card_data.get("scenario", ""),
                    first_message=card_data.get("first_mes", ""),
                    example_messages=card_data.get("mes_example", ""),
                    system_prompt=card_data.get("system_prompt", ""),
                )

        if chunk_type == b"IEND":
            break

    raise ValueError(f"No character card found in PNG: {path}")


class CharacterCard(ConstantComponent[tuple[()], CharacterCardOutputs]):
    """A constant component holding a character card (persona definition).

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
    def get_config_options(
        cls, field: str, values: dict[str, Any] | None = None
    ) -> list[dict[str, str]] | None:
        if field != "path":
            return None
        if not cls._PRESETS_DIR.exists():
            return None
        return [
            {"value": str(p), "label": p.stem.replace("_", " ").title()}
            for p in sorted(cls._PRESETS_DIR.glob("*.png"))
        ]

    def __init__(
        self,
        path: Path | None = None,
        config: CharacterCardConfig | None = None,
    ) -> None:
        super().__init__()
        if path is not None:
            self.config = _read_png_character_card(path)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("CharacterCard requires either path or config")

    def get_values(self) -> CharacterCardOutputs:
        return CharacterCardOutputs(
            name=TextFrame.new(text=self.config.name),
            description=TextFrame.new(text=self.config.description),
            personality=TextFrame.new(text=self.config.personality),
            scenario=TextFrame.new(text=self.config.scenario),
            first_message=TextFrame.new(text=self.config.first_message),
            example_messages=TextFrame.new(text=self.config.example_messages),
            system_prompt=TextFrame.new(text=self.config.system_prompt),
        )

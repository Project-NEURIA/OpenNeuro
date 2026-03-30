from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.conduit.character_card import (
    CharacterCard,
    CharacterCardConfig,
    CharacterCardOutputs,
)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + b"\x00\x00\x00\x00"


def _write_card_png(path: Path, keyword: str = "chara") -> CharacterCardConfig:
    payload = {
        "data": {
            "name": "Alice",
            "description": "A careful tester",
            "personality": "Direct",
            "scenario": "Unit tests",
            "first_mes": "Hi",
            "mes_example": "Example dialog",
            "system_prompt": "Be precise",
            "post_history_instructions": "Stay concise",
        }
    }
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8"))
    text_data = keyword.encode("latin-1") + b"\x00" + encoded
    png = (
        b"\x89PNG\r\n\x1a\n" + _png_chunk(b"tEXt", text_data) + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return CharacterCardConfig(
        name="Alice",
        description="A careful tester",
        personality="Direct",
        scenario="Unit tests",
        first_message="Hi",
        example_messages="Example dialog",
        system_prompt="Be precise",
        post_history_instructions="Stay concise",
    )


def test_character_card_png_and_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    png_path = tmp_path / "card.png"
    expected = _write_card_png(png_path)

    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    preset_path = preset_dir / "cool_mature_woman.png"
    _write_card_png(preset_path, keyword="ccv3")
    monkeypatch.setattr(CharacterCard, "_PRESETS_DIR", preset_dir)

    assert CharacterCard.get_options({})["preset"] == CharacterCard._PRESET_OPTIONS
    assert CharacterCard._resolve_preset("cool_mature_woman") == preset_path
    assert CharacterCard._read_png(png_path) == expected

    with pytest.raises(ValueError, match="Preset not found"):
        CharacterCard._resolve_preset("missing")

    bad_png = tmp_path / "bad.png"
    bad_png.write_bytes(b"not-a-png")
    with pytest.raises(ValueError, match="Not a valid PNG"):
        CharacterCard._read_png(bad_png)

    empty_png = tmp_path / "empty.png"
    empty_png.write_bytes(b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IEND", b""))
    with pytest.raises(ValueError, match="No character card found"):
        CharacterCard._read_png(empty_png)

    preset_card = CharacterCard(preset="cool_mature_woman")
    path_card = CharacterCard(path=png_path)
    config_card = CharacterCard(config=expected)
    assert preset_card.config.name == "Alice"
    assert path_card.config.system_prompt == "Be precise"
    assert config_card.config.personality == "Direct"

    with pytest.raises(ValueError, match="requires preset, path, or config"):
        CharacterCard()

    sent: dict[str, list[object]] = {
        "prompts": [],
        "system_prompt": [],
        "name": [],
        "description": [],
        "personality": [],
        "scenario": [],
        "first_message": [],
        "example_messages": [],
        "post_history_instructions": [],
    }
    outputs = CharacterCardOutputs(
        prompts=SimpleNamespace(send=lambda value: sent["prompts"].append(value)),
        system_prompt=SimpleNamespace(
            send=lambda value: sent["system_prompt"].append(value)
        ),
        name=SimpleNamespace(send=lambda value: sent["name"].append(value)),
        description=SimpleNamespace(
            send=lambda value: sent["description"].append(value)
        ),
        personality=SimpleNamespace(
            send=lambda value: sent["personality"].append(value)
        ),
        scenario=SimpleNamespace(send=lambda value: sent["scenario"].append(value)),
        first_message=SimpleNamespace(
            send=lambda value: sent["first_message"].append(value)
        ),
        example_messages=SimpleNamespace(
            send=lambda value: sent["example_messages"].append(value)
        ),
        post_history_instructions=SimpleNamespace(
            send=lambda value: sent["post_history_instructions"].append(value)
        ),
    )

    config_card.setup(outputs)

    prompts = sent["prompts"][0]
    assert [frame.content for frame in prompts] == [
        "A careful tester",
        "Direct",
        "Unit tests",
        "Example dialog",
        "Be precise",
    ]
    assert sent["system_prompt"][0].get() == "Be precise"
    assert sent["name"][0].get() == "Alice"
    assert sent["description"][0].get() == "A careful tester"
    assert sent["personality"][0].get() == "Direct"
    assert sent["scenario"][0].get() == "Unit tests"
    assert sent["first_message"][0].get() == "Hi"
    assert sent["example_messages"][0].get() == "Example dialog"
    assert sent["post_history_instructions"][0].get() == "Stay concise"

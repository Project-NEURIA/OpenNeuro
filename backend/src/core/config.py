from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    def save_json(self, path: str | Path) -> None:
        """Saves the config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def from_json(cls: Type[T], path: str | Path) -> T:
        """Loads the config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """Loads the config from a dictionary."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Converts the config to a dictionary."""
        return self.model_dump()

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return self.model_dump_json(indent=2)

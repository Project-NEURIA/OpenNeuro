import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel


BASE_DIR = Path.home() / "Documents" / "OpenNeuro"
PROJECTS_DIR = BASE_DIR / "projects"
CONFIG_PATH = BASE_DIR / "config.json"


class AppConfig(BaseModel):
    current_project: str | None = None

    @classmethod
    def load_config(cls) -> Self:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return cls.model_validate(data)
        return cls()

    def save_config(self) -> None:
        CONFIG_PATH.write_text(self.model_dump_json(indent=2))

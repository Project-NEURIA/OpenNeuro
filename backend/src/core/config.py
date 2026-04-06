import json
import os
from pathlib import Path
from typing import Self

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

BASE_DIR = Path.home() / "Documents" / "OpenNeuro"

DEPLOY_MODE: str = os.environ.get("DEPLOY_MODE", "local")
PCVR_IP: str = os.environ.get("PCVR_IP", "127.0.0.1")
UPLOADS_DIR = BASE_DIR / "uploads"

PROJECTS_DIR = BASE_DIR / "projects"
CONFIG_PATH = BASE_DIR / "config.json"
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
PRESETS_DIR = ASSETS_DIR / "presets"


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

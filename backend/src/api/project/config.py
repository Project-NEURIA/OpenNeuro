from __future__ import annotations

import json

from pydantic import BaseModel

from src.api.project.paths import CONFIG_PATH


class AppConfig(BaseModel):
    current_project: str | None = None


def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        return AppConfig.model_validate(data)
    return AppConfig()


def save_config(config: AppConfig) -> None:
    CONFIG_PATH.write_text(config.model_dump_json(indent=2))

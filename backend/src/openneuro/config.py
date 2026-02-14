"""Global config management for OpenNeuro backend."""

import json
from pathlib import Path
from typing import Any

# Global config
_config: dict[str, Any] = {}


def load_config() -> dict[str, Any]:
    """Load chat config from file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "chat_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            _config.update(json.load(f))
            print(f"[config] Loaded config with keys: {list(_config.keys())}")
            return _config
    return {}


def get_config() -> dict[str, Any]:
    """Get the current config."""
    return _config


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific config value by key (supports nested keys with dots)."""
    keys = key.split(".")
    value = _config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value

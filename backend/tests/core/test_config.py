from __future__ import annotations

from src.core import config


def test_load_config_missing_file(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    loaded = config.AppConfig.load_config()
    assert loaded.current_project is None


def test_load_and_save_config(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "CONFIG_PATH", cfg_path)
    cfg_path.write_text('{"current_project":"demo"}')
    loaded = config.AppConfig.load_config()
    assert loaded.current_project == "demo"
    loaded.current_project = "next"
    loaded.save_config()
    assert "next" in cfg_path.read_text()

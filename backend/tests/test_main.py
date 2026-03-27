from __future__ import annotations

import asyncio
import sys
import types

import src.main as main_module


def test_start_parent_watchdog_starts_daemon_thread(monkeypatch) -> None:
    started = {}
    ppid_values = iter([1234, 9999])

    class FakeThread:
        def __init__(self, target, daemon):
            started["target"] = target
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(main_module, "_start_parent_watchdog", main_module._start_parent_watchdog)
    monkeypatch.setattr("os.getppid", lambda: next(ppid_values))
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr("os._exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    monkeypatch.setattr("threading.Thread", FakeThread)
    main_module._start_parent_watchdog()
    assert started["daemon"] is True
    assert started["started"] is True
    try:
        started["target"]()
    except SystemExit as e:
        assert e.code == 0


def test_main_invokes_uvicorn_and_env(monkeypatch) -> None:
    called = {}

    monkeypatch.setattr(main_module, "load_dotenv", lambda *a, **k: called.setdefault("dotenv", True))
    monkeypatch.setattr(main_module, "_start_parent_watchdog", lambda: called.setdefault("watchdog", True))

    fake_uvicorn = types.SimpleNamespace(
        run=lambda app, host, port, log_level: called.setdefault("uvicorn", (host, port, log_level))
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    main_module.main()
    assert called["dotenv"] is True
    assert called["watchdog"] is True
    assert called["uvicorn"] == ("0.0.0.0", 8000, "warning")


def test_lifespan_sets_state(monkeypatch, tmp_path) -> None:
    class DummyConfig:
        current_project = "demo"

    called = {}
    monkeypatch.setattr(main_module, "_copy_presets", lambda: called.setdefault("copy", True))
    monkeypatch.setattr(main_module, "install_global_stream_capture", lambda: called.setdefault("capture", True))
    monkeypatch.setattr(main_module.AppConfig, "load_config", classmethod(lambda cls: DummyConfig()))
    monkeypatch.setattr(main_module, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(main_module, "GraphManager", lambda graph: ("manager", graph))

    app = types.SimpleNamespace(state=types.SimpleNamespace())

    async def run_lifespan():
        async with main_module.lifespan(app):
            assert app.state.current_project == "demo"
            assert app.state.manager[0] == "manager"

    asyncio.run(run_lifespan())
    assert called["copy"] is True
    assert called["capture"] is True

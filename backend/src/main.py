from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.graph.node.controller import router as node_router
from src.api.graph.edge.controller import router as edge_router
from src.api.graph.run.controller import router as run_router
from src.api.graph.save.controller import router as save_router
from src.api.metrics.controller import router as metrics_router
from src.api.logs.controller import router as logs_router
from src.api.component.controller import router as component_router
from src.api.project.controller import router as project_router
from src.api.ui.controller import router as ui_router
from src.api.env.controller import router as env_router
from src.api.upload.controller import router as upload_router
from src.api.ui.bridge import UIChannelBridge
from src.core.graph import Graph
from src.core.config import DEPLOY_MODE, PROJECTS_DIR, PRESETS_DIR, AppConfig
from src.core.graph import GraphManager
from src.core.log_capture import install_global_stream_capture


def _copy_presets() -> None:
    """Copy bundled presets into the user's projects directory (skip existing)."""
    if not PRESETS_DIR.exists():
        return
    for preset in PRESETS_DIR.iterdir():
        if not preset.is_dir():
            continue
        dest = PROJECTS_DIR / preset.name
        if dest.exists():
            continue
        import shutil

        shutil.copytree(preset, dest)


@asynccontextmanager
async def lifespan(app: FastAPI):
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    _copy_presets()
    install_global_stream_capture()

    config = AppConfig.load_config()
    app.state.current_project = config.current_project

    print("[backend] Registering components...")
    app.state.manager = GraphManager(Graph(edges=[], nodes={}))
    app.state.ui_bridge = UIChannelBridge()
    print("[backend] Ready")

    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(node_router)
app.include_router(edge_router)
app.include_router(run_router)
app.include_router(save_router)
app.include_router(metrics_router)
app.include_router(logs_router)
app.include_router(component_router)
app.include_router(project_router)
app.include_router(ui_router)
app.include_router(env_router)
app.include_router(upload_router)

# Serve built frontend as static files when running remotely.
# Must come after all include_router calls so API routes match first.
if DEPLOY_MODE == "remote":
    _frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if _frontend_dist.exists():
        from starlette.staticfiles import StaticFiles

        app.mount(
            "/",
            StaticFiles(directory=str(_frontend_dist), html=True),
            name="frontend",
        )


def _start_parent_watchdog() -> None:
    """Exit if parent process dies (e.g. Tauri/bun killed by Ctrl+C)."""
    import os
    import threading

    ppid = os.getppid()

    def watch() -> None:
        import time

        while True:
            time.sleep(1)
            if os.getppid() != ppid:
                os._exit(0)

    t = threading.Thread(target=watch, daemon=True)
    t.start()


def main() -> None:
    if DEPLOY_MODE != "remote":
        _start_parent_watchdog()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    import uvicorn

    print(f"[backend] API server starting on http://{host}:{port} (mode={DEPLOY_MODE})")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from src.api.project import service
from src.api.project.config import AppConfig
from src.api.project.dto import (
    CurrentProjectResponse,
    ProjectCreateRequest,
    ProjectSummary,
)
from src.api.project.paths import thumbnail_path

router = APIRouter()


def _get_config(request: Request) -> AppConfig:
    return request.app.state.config  # type: ignore[no-any-return]


@router.get("/projects")
def list_projects() -> list[ProjectSummary]:
    return service.list_projects()


@router.post("/projects", status_code=201)
def create_project(req: ProjectCreateRequest) -> None:
    service.create_project(req.name)


@router.delete("/projects/{name}", status_code=204)
def delete_project(name: str, request: Request) -> None:
    config = _get_config(request)
    if config.current_project == name:
        raise HTTPException(
            status_code=400, detail="Cannot delete the currently open project"
        )
    service.delete_project(name)


@router.post("/projects/{name}/start")
def start_project(name: str, request: Request) -> CurrentProjectResponse:
    config = _get_config(request)
    current_graph = getattr(request.app.state, "graph", None)
    graph = service.start_project(name, config, current_graph)
    request.app.state.graph = graph
    request.app.state.current_project = name
    return CurrentProjectResponse(current_project=name)


@router.post("/project/close", status_code=204)
def close_project(request: Request) -> None:
    config = _get_config(request)
    current_graph = getattr(request.app.state, "graph", None)
    service.close_project(config, current_graph)
    from src.api.graph.domain.graph import Graph

    request.app.state.graph = Graph(edges=[], nodes={})
    request.app.state.current_project = None


@router.get("/project/current")
def get_current_project(request: Request) -> CurrentProjectResponse:
    config = _get_config(request)
    return CurrentProjectResponse(current_project=config.current_project)


@router.get("/projects/{name}/thumbnail")
def get_thumbnail(name: str) -> FileResponse:
    path = thumbnail_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path, media_type="image/png")

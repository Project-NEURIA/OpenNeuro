from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from src.api.dep import get_manager
from src.api.project import service
from src.api.project.dto import (
    CurrentProjectResponse,
    ProjectCreateRequest,
    ProjectSummary,
)
from src.core.config import PROJECTS_DIR, AppConfig
from src.core.graph import GraphManager

router = APIRouter()


@router.get("/projects")
def list_projects() -> list[ProjectSummary]:
    return service.list_projects()


@router.post("/projects", status_code=201)
def create_project(req: ProjectCreateRequest) -> None:
    service.create_project(req.name)


@router.delete("/projects/{name}", status_code=204)
def delete_project(name: str) -> None:
    service.delete_project(name)


@router.post("/projects/{name}/start")
def start_project(
    name: str, manager: GraphManager = Depends(get_manager)
) -> CurrentProjectResponse:
    service.start_project(name, manager)
    return CurrentProjectResponse(current_project=name)


@router.post("/project/close", status_code=204)
def close_project(manager: GraphManager = Depends(get_manager)) -> None:
    service.close_project(manager)


@router.get("/project/current")
def get_current_project() -> CurrentProjectResponse:
    config = AppConfig.load_config()
    return CurrentProjectResponse(current_project=config.current_project)


@router.get("/projects/{name}/thumbnail")
def get_thumbnail(name: str) -> FileResponse:
    path = PROJECTS_DIR / name / "thumbnail.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(path, media_type="image/png")

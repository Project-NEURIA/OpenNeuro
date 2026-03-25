from __future__ import annotations

from pydantic import BaseModel


class ProjectSummary(BaseModel):
    name: str
    has_thumbnail: bool


class ProjectCreateRequest(BaseModel):
    name: str


class CurrentProjectResponse(BaseModel):
    current_project: str | None


class ProjectExportRequest(BaseModel):
    name: str


class ProjectExportResponse(BaseModel):
    project_dir: str
    assets_copied: list[str]


class ProjectImportRequest(BaseModel):
    git_url: str


class ProjectImportResponse(BaseModel):
    name: str

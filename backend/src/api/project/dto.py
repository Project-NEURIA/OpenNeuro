from __future__ import annotations

from pydantic import BaseModel


class ProjectSummary(BaseModel):
    name: str
    has_thumbnail: bool


class ProjectCreateRequest(BaseModel):
    name: str


class CurrentProjectResponse(BaseModel):
    current_project: str | None

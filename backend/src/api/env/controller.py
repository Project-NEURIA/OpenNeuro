from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.core.config import BASE_DIR

router = APIRouter()

ENV_PATH = BASE_DIR / ".env"


class EnvBody(BaseModel):
    content: str


@router.get("/env")
def get_env() -> EnvBody:
    if ENV_PATH.exists():
        return EnvBody(content=ENV_PATH.read_text())
    return EnvBody(content="")


@router.put("/env")
def put_env(body: EnvBody) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text(body.content)

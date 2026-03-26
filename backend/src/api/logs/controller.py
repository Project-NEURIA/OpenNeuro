from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.core.log_capture import get_log_store

router = APIRouter(prefix="/logs")


class ComponentLogEntry(BaseModel):
    seq: int
    timestamp: float
    stream: Literal["stdout", "stderr"]
    text: str


class ComponentLogsResponse(BaseModel):
    node_id: str
    entries: list[ComponentLogEntry]


@router.get("/{node_id}")
def get_component_logs(
    node_id: str,
    after: int = Query(0, ge=0),
    limit: int = Query(400, ge=1, le=2000),
) -> ComponentLogsResponse:
    entries = get_log_store().get_entries(node_id, after=after, limit=limit)
    return ComponentLogsResponse(
        node_id=node_id,
        entries=[
            ComponentLogEntry(
                seq=entry.seq,
                timestamp=entry.timestamp,
                stream=cast(Literal["stdout", "stderr"], entry.stream),
                text=entry.text,
            )
            for entry in entries
        ],
    )

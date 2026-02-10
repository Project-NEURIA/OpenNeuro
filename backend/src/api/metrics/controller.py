from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from ...core.graph import Graph
from ...core.component import Component
from ..dep import get_graph
from . import service

router = APIRouter(prefix="/metrics")


async def _stream(graph: Graph[Component]) -> AsyncGenerator[str, None]:
    while True:
        yield service.collect(graph).model_dump_json()
        await asyncio.sleep(0.1)


@router.get("")
async def get_metrics(graph: Graph[Component] = Depends(get_graph)) -> EventSourceResponse:
    return EventSourceResponse(_stream(graph))

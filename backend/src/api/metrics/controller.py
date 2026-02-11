from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.api.graph.domain.graph import Graph
from src.core.component import Component
from src.api.dep import get_graph
from src.api.metrics import service

router = APIRouter(prefix="/metrics")


async def _stream(graph: Graph[Component]) -> AsyncGenerator[str, None]:
    while True:
        yield service.collect(graph).model_dump_json()
        await asyncio.sleep(0.1)


@router.get("")
async def get_metrics(graph: Graph[Component] = Depends(get_graph)) -> EventSourceResponse:
    return EventSourceResponse(_stream(graph))

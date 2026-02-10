from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from ...core.graph import Graph
from ...core.component import Component
from .dto import MetricsResponse
from . import service

router = APIRouter(prefix="/metrics")


def get_graph(request: Request) -> Graph[Component]:
    return request.app.state.graph


async def _stream(graph: Graph[Component]) -> AsyncGenerator[str, None]:
    while True:
        raw = service.collect(graph)
        data = MetricsResponse(**raw)
        yield data.model_dump_json()
        await asyncio.sleep(1)


@router.get("")
async def get_metrics(graph: Graph[Component] = Depends(get_graph)) -> EventSourceResponse:
    return EventSourceResponse(_stream(graph))

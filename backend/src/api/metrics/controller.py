from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.api.dep import get_manager
from src.api.metrics.service import MetricsCollector
from src.core.graph import GraphManager

router = APIRouter(prefix="/metrics")


async def _stream(manager: GraphManager) -> AsyncGenerator[str, None]:
    collector = MetricsCollector()
    while True:
        yield collector.collect(manager).model_dump_json()
        await asyncio.sleep(0.1)


@router.get("")
async def get_metrics(
    manager: GraphManager = Depends(get_manager),
) -> EventSourceResponse:
    return EventSourceResponse(_stream(manager))

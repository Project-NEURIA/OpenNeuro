from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from openneuro.api.frames import service

router = APIRouter(prefix="/frames")


async def _stream() -> AsyncGenerator[str, None]:
    while True:
        yield service.collect().model_dump_json()
        await asyncio.sleep(0.1)


@router.get("")
async def get_frames() -> EventSourceResponse:
    return EventSourceResponse(_stream())

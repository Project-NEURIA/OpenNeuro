from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from src.core.config import UPLOADS_DIR

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a file upload and return its server-side path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix if file.filename else ""
    dest = UPLOADS_DIR / f"{uuid.uuid4()}{ext}"
    dest.write_bytes(await file.read())
    return {"path": str(dest)}

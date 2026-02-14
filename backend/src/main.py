from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.graph.controller import router as graph_router
from src.api.metrics.controller import router as metrics_router
from src.api.component.controller import router as component_router
from src.api.video.controller import router as video_router
from src.api.graph.domain.graph import Graph


from src.api.graph import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = service.load_graph()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graph_router)
app.include_router(metrics_router)
app.include_router(component_router)
app.include_router(video_router)


def main() -> None:
    load_dotenv(override=True)

    import uvicorn
    print("[backend] API server starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()

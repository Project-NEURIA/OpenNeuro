from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.graph.node.controller import router as node_router
from src.api.graph.edge.controller import router as edge_router
from src.api.graph.run.controller import router as run_router
from src.api.graph.save import service as save_service
from src.api.metrics.controller import router as metrics_router
from src.api.component.controller import router as component_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = save_service.load_graph()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(node_router)
app.include_router(edge_router)
app.include_router(run_router)
app.include_router(metrics_router)
app.include_router(component_router)


def main() -> None:
    load_dotenv(override=True)

    import uvicorn

    print("[backend] API server starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core.node import Node, _NONE
from .metrics import MetricsCollector
from .pipeline import PipelineManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_collector = MetricsCollector()
manager = PipelineManager()

# Static type map — generics are erased at runtime, so we label manually.
_TYPE_LABELS: dict[str, tuple[str | None, str | None]] = {
    "Mic": (None, "bytes"),
    "VAD":        ("bytes", "bytes"),
    "ASR":        ("bytes", "str"),
    "LLM":        ("str", "str"),
    "TTS":        ("str", "bytes"),
    "STS":        ("bytes", "bytes"),
    "Speaker":    ("bytes", None),
}


def _build_topology() -> dict:
    """Inspect registries to produce {nodes, edges} for the frontend."""
    # Map each Topic to its owning node (the node whose output topic it is)
    topic_to_owner: dict[int, Node] = {}
    for node in Node._registry:
        out = node.topic
        if out is not _NONE:
            topic_to_owner[id(out)] = node

    nodes = []
    edges = []

    for node in Node._registry:
        has_input = node.input is not _NONE
        has_output = node.topic is not _NONE

        if not has_input:
            category = "source"
        elif not has_output:
            category = "sink"
        else:
            category = "conduit"

        type_info = _TYPE_LABELS.get(node.name, (None, None))

        nodes.append({
            "id": node.name,
            "name": node.name,
            "category": category,
            "input_type": type_info[0],
            "output_type": type_info[1],
            "status": node._status.value,
        })

        # Derive edge: if this node has an input topic, find the owning node
        if has_input:
            input_topic = node.input
            source_node = topic_to_owner.get(id(input_topic))
            if source_node is not None:
                edges.append({
                    "id": f"{source_node.name}->{node.name}",
                    "source": source_node.name,
                    "target": node.name,
                    "topic_name": input_topic.name,
                })

    return {"nodes": nodes, "edges": edges}


class PipelineConfig(BaseModel):
    nodes: list[str]
    edges: list[dict[str, str]]


@app.get("/api/pipeline")
def get_pipeline() -> dict:
    return _build_topology()


@app.post("/api/pipeline")
def post_pipeline(config: PipelineConfig) -> dict[str, str]:
    return manager.apply({"nodes": config.nodes, "edges": config.edges})


@app.websocket("/api/ws/metrics")
async def ws_metrics(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            data = _collector.collect()
            await ws.send_json(data)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

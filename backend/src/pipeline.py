from __future__ import annotations

import threading
from typing import Any

from .core.node import Node, _NONE
from .core.topic import Topic
from .core.source import Microphone
from .core.sink import Speaker
from .core.conduit import VAD, ASR, LLM, TTS, STS

NODE_CLASSES: dict[str, type[Node]] = {
    "Microphone": Microphone,
    "VAD": VAD,
    "ASR": ASR,
    "LLM": LLM,
    "TTS": TTS,
    "STS": STS,
    "Speaker": Speaker,
}

# Which nodes are sources (no input stream needed)
_SOURCES = {"Microphone"}
# Which nodes are sinks (no output topic)
_SINKS = {"Speaker"}


class PipelineManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: list[Node] = []

    def apply(self, config: dict[str, Any]) -> dict[str, str]:
        """Rebuild the pipeline from a {nodes, edges} config.

        Returns {"status": "ok"} on success or {"status": "error", "detail": ...}.
        """
        with self._lock:
            return self._apply(config)

    def _apply(self, config: dict[str, Any]) -> dict[str, str]:
        # --- Stop the old pipeline ---
        for node in self._nodes:
            try:
                node.stop()
            except Exception:
                pass
        self._nodes.clear()

        # Clear any stragglers from registries
        Node._registry.clear()
        Topic._registry.clear()

        # --- Parse config ---
        node_names: list[str] = config.get("nodes", [])
        edges: list[dict[str, str]] = config.get("edges", [])

        # Validate node names
        for name in node_names:
            if name not in NODE_CLASSES:
                return {"status": "error", "detail": f"Unknown node: {name}"}

        # Build adjacency: source_name -> list[target_name]
        adj: dict[str, list[str]] = {n: [] for n in node_names}
        in_edges: dict[str, str] = {}  # target -> source
        for edge in edges:
            src, tgt = edge["source"], edge["target"]
            if src not in adj or tgt not in adj:
                return {"status": "error", "detail": f"Edge references unknown node: {src}->{tgt}"}
            adj[src].append(tgt)
            in_edges[tgt] = src

        # --- Topological sort ---
        in_degree: dict[str, int] = {n: 0 for n in node_names}
        for tgt in in_edges:
            in_degree[tgt] += 1

        queue: list[str] = [n for n in node_names if in_degree[n] == 0]
        topo_order: list[str] = []
        while queue:
            cur = queue.pop(0)
            topo_order.append(cur)
            for nxt in adj[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(topo_order) != len(node_names):
            return {"status": "error", "detail": "Cycle detected in pipeline graph"}

        # --- Instantiate nodes in topo order ---
        instances: dict[str, Node] = {}

        for name in topo_order:
            cls = NODE_CLASSES[name]

            if name in _SOURCES:
                # Source nodes: no input stream
                node = cls()
            elif name in _SINKS:
                # Sink nodes: need input stream from upstream
                upstream = in_edges.get(name)
                if upstream is None or upstream not in instances:
                    return {"status": "error", "detail": f"Sink {name} has no connected source"}
                stream = instances[upstream].topic.stream()
                node = cls(stream)
            else:
                # Conduit nodes: need input stream from upstream
                upstream = in_edges.get(name)
                if upstream is None or upstream not in instances:
                    return {"status": "error", "detail": f"Conduit {name} has no connected source"}
                stream = instances[upstream].topic.stream()
                node = cls(stream)

            instances[name] = node

        # Name output topics
        for name, node in instances.items():
            if node.topic is not _NONE:
                node.topic.name = f"{name}_out"

        # --- Start all nodes ---
        self._nodes = list(instances.values())
        for node in self._nodes:
            node.start()

        return {"status": "ok"}

from __future__ import annotations

import time

from .core.topic import Topic
from .core.node import Node


class MetricsCollector:
    def __init__(self) -> None:
        self._prev_counts: dict[str, int] = {}
        self._prev_time: float = time.time()

    def collect(self) -> dict:
        now = time.time()
        dt = now - self._prev_time
        if dt <= 0:
            dt = 1e-9
        self._prev_time = now

        topics = {}
        for t in Topic.all_topics():
            snap = t.snapshot()
            prev = self._prev_counts.get(snap["name"], 0)
            rate = (snap["msg_count"] - prev) / dt
            self._prev_counts[snap["name"]] = snap["msg_count"]
            snap["msg_per_sec"] = round(rate, 2)
            topics[snap["name"]] = snap

        nodes = {}
        for n in Node._registry:
            meta = n.metadata()
            nodes[meta["name"]] = meta

        return {
            "topics": topics,
            "nodes": nodes,
            "timestamp": now,
        }

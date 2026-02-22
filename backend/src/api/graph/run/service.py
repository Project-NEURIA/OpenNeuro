from __future__ import annotations

from src.core.graph import GraphManager


def start_all(manager: GraphManager) -> None:
    manager.run()


def stop_all(manager: GraphManager) -> None:
    manager.stop()

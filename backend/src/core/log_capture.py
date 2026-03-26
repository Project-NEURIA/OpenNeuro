from __future__ import annotations

import io
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class LogEntry:
    seq: int
    timestamp: float
    stream: str
    text: str


class ComponentLogStore:
    def __init__(self, max_entries_per_node: int = 1000) -> None:
        self._max_entries_per_node = max_entries_per_node
        self._entries: dict[str, deque[LogEntry]] = {}
        self._thread_to_node: dict[int, str] = {}
        self._partials: dict[tuple[int, str], str] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def register_thread(self, node_id: str) -> None:
        ident = threading.get_ident()
        with self._lock:
            self._thread_to_node[ident] = node_id

    def unregister_thread(self) -> None:
        ident = threading.get_ident()
        with self._lock:
            node_id = self._thread_to_node.pop(ident, None)
            if node_id is None:
                return
            for stream in ("stdout", "stderr"):
                self._flush_partial_locked(ident, node_id, stream)

    def append_from_thread(self, stream: str, text: str) -> None:
        ident = threading.get_ident()
        with self._lock:
            node_id = self._thread_to_node.get(ident)
            if not node_id:
                return

            key = (ident, stream)
            current = self._partials.get(key, "")
            current += text

            while True:
                newline_idx = current.find("\n")
                if newline_idx < 0:
                    break
                line = current[:newline_idx].rstrip("\r")
                self._append_locked(node_id, stream, line)
                current = current[newline_idx + 1 :]

            self._partials[key] = current

    def get_entries(
        self, node_id: str, after: int = 0, limit: int = 400
    ) -> list[LogEntry]:
        with self._lock:
            entries = list(self._entries.get(node_id, ()))
        if after > 0:
            entries = [entry for entry in entries if entry.seq > after]
        if len(entries) > limit:
            entries = entries[-limit:]
        return entries

    def clear_node(self, node_id: str) -> None:
        with self._lock:
            self._entries.pop(node_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._entries.clear()

    def _append_locked(self, node_id: str, stream: str, text: str) -> None:
        self._seq += 1
        bucket = self._entries.get(node_id)
        if bucket is None:
            bucket = deque(maxlen=self._max_entries_per_node)
            self._entries[node_id] = bucket
        bucket.append(
            LogEntry(
                seq=self._seq,
                timestamp=time.time(),
                stream=stream,
                text=text,
            )
        )

    def _flush_partial_locked(self, ident: int, node_id: str, stream: str) -> None:
        key = (ident, stream)
        partial = self._partials.pop(key, "")
        if partial:
            self._append_locked(node_id, stream, partial.rstrip("\r"))


class _RoutedStream(io.TextIOBase):
    def __init__(
        self, stream_name: str, original: TextIO, store: ComponentLogStore
    ) -> None:
        self._stream_name = stream_name
        self._original = original
        self._store = store

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._store.append_from_thread(self._stream_name, s)
        return self._original.write(s)

    def flush(self) -> None:
        self._original.flush()

    def isatty(self) -> bool:
        return self._original.isatty()


_store = ComponentLogStore()
_installed = False


def get_log_store() -> ComponentLogStore:
    return _store


def install_global_stream_capture() -> None:
    global _installed
    if _installed:
        return
    sys.stdout = _RoutedStream("stdout", sys.stdout, _store)
    sys.stderr = _RoutedStream("stderr", sys.stderr, _store)
    _installed = True

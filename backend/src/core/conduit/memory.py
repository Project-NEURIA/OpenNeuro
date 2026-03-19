from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import MessageFrame, TextFrame

from mem0 import Memory  # type: ignore[import-untyped]


class Mem0VectorStoreConfig(BaseModel):
    provider: str = "qdrant"
    host: str | None = None
    port: int | None = None
    path: str | None = None
    url: str | None = None
    api_key_env_var: str | None = None
    on_disk: bool = True
    collection_name: str = "mem0_conversations"
    embedding_model_dims: int = 1536


class Mem0Config(BaseModel):
    user_id: str = "default"
    last_k: int = 4
    memory_limit: int = 5

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1-nano-2025-04-14"
    llm_api_key_env_var: str = "OPENAI_API_KEY"
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 400

    embedder_provider: str = "openai"
    embedder_model: str = "text-embedding-3-small"
    embedder_api_key_env_var: str = "OPENAI_API_KEY"
    embedder_base_url: str | None = None
    embedder_dims: int = 1536

    vector_store: Mem0VectorStoreConfig | None = None


class Mem0Inputs(NamedTuple):
    messages: Receiver[list[MessageFrame]]


class Mem0Outputs(NamedTuple):
    memory_prefix: Sender[TextFrame]


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise ValueError(f"Environment variable {var} must be set")
    return val


def _build_mem0_config(config: Mem0Config) -> dict[str, Any]:
    """Build the mem0 Memory config dict from our pydantic config."""
    cfg: dict[str, Any] = {
        "llm": {
            "provider": config.llm_provider,
            "config": {
                "model": config.llm_model,
                "api_key": _require_env(config.llm_api_key_env_var),
                "temperature": config.llm_temperature,
                "max_tokens": config.llm_max_tokens,
                **(
                    {"openai_base_url": config.llm_base_url}
                    if config.llm_base_url
                    else {}
                ),
            },
        },
        "embedder": {
            "provider": config.embedder_provider,
            "config": {
                "model": config.embedder_model,
                "api_key": _require_env(config.embedder_api_key_env_var),
                "embedding_dims": config.embedder_dims,
                **(
                    {"openai_base_url": config.embedder_base_url}
                    if config.embedder_base_url
                    else {}
                ),
            },
        },
    }

    # Vector store — always include so we control on_disk / path.
    vs = config.vector_store or Mem0VectorStoreConfig()
    vs_cfg: dict[str, Any] = {
        "collection_name": vs.collection_name,
        "embedding_model_dims": vs.embedding_model_dims,
        "on_disk": vs.on_disk,
    }
    if vs.host and vs.port:
        vs_cfg["host"] = vs.host
        vs_cfg["port"] = vs.port
    elif vs.url:
        vs_cfg["url"] = vs.url
    else:
        vs_cfg["path"] = vs.path or str(Path.home() / ".mem0" / "qdrant")
        os.makedirs(vs_cfg["path"], exist_ok=True)
    if vs.api_key_env_var:
        vs_cfg["api_key"] = _require_env(vs.api_key_env_var)

    cfg["vector_store"] = {"provider": vs.provider, "config": vs_cfg}
    return cfg


def _format_memory_prefix(results: list[dict[str, Any]]) -> str:
    """Format mem0 search results into a memory prefix string."""
    memories = [r["memory"] for r in results if r.get("memory")]
    if not memories:
        return ""
    body = "\n".join(f"- {m}" for m in memories)
    return f"[Relevant memories]\n{body}\n[End of memories]"


# Module-level cache: local Qdrant only allows one client on a given path,
# so we reuse the Memory instance across graph resets.
_memory_instance: Memory | None = None
_memory_config_hash: int | None = None


def _close_memory(mem: Memory) -> None:
    """Release file locks held by a Memory instance's local Qdrant clients.

    mem0.Memory.__init__ creates two Qdrant vector stores:
      - mem.vector_store          (mem0.vector_stores.qdrant.Qdrant)  — user data
      - mem._telemetry_vector_store  (same type)                      — migrations/telemetry
    Each has a `.client` (qdrant_client.QdrantClient) whose `.close()` releases
    the portalocker file lock on the local SQLite storage.
    """
    mem.vector_store.client.close()
    mem._telemetry_vector_store.client.close()


def _get_or_create_memory(config: Mem0Config) -> Memory:
    global _memory_instance, _memory_config_hash
    cfg_hash = hash(config.model_dump_json())
    if _memory_instance is not None:
        if cfg_hash == _memory_config_hash:
            return _memory_instance
        _close_memory(_memory_instance)
    _memory_instance = Memory.from_config(_build_mem0_config(config))
    _memory_config_hash = cfg_hash
    return _memory_instance


class Mem0(ThreadedComponent[Mem0Inputs, Mem0Outputs]):
    description = "Stores and retrieves conversational memory"

    """Memory component backed by mem0.

    On each incoming list[MessageFrame]:
      1. Query memory using the last_k turns as search context.
      2. Send the formatted memory prefix back as a TextFrame.
      3. Asynchronously update memory with the new conversation turns.
    """

    tags = Tag(io={"conduit"}, functionality={"llm"})

    def __init__(self, config: Mem0Config) -> None:
        super().__init__()
        self.config = config
        self._prev_msg_count: int = 0
        self._memory = _get_or_create_memory(config)
        print("[Mem0] Memory initialised")

    def _query_memory(self, turns: list[MessageFrame]) -> str:
        if not turns:
            return ""
        query = "\n".join(f"{m.role}: {m.content}" for m in turns if m.content)
        try:
            results = self._memory.search(
                query=query,
                user_id=self.config.user_id,
                limit=self.config.memory_limit,
            )
            hits = results.get("results", []) if isinstance(results, dict) else results
            return _format_memory_prefix(hits)
        except Exception as e:
            print(f"[Mem0] Search error: {e}")
            return ""

    def _update_memory(self, new_messages: list[MessageFrame]) -> None:
        try:
            self._memory.add(
                [{"role": m.role, "content": m.content} for m in new_messages],
                user_id=self.config.user_id,
            )
        except Exception as e:
            print(f"[Mem0] Add error: {e}")

    def run(self, inputs: Mem0Inputs, outputs: Mem0Outputs) -> None:
        print("[Mem0] Starting Memory component")

        for frame in inputs.messages(self):
            if frame is None:
                break
            if not frame:
                continue

            # 1. Query memory with the last k turns
            non_system = [m for m in frame if m.role != "system"]
            recent = non_system[-(self.config.last_k * 2) :]
            prefix = self._query_memory(recent)

            # 2. Send memory prefix (even if empty, so downstream isn't stuck)
            outputs.memory_prefix.send(TextFrame.new(text=prefix))

            # 3. Update memory in background with only the new messages
            new_messages = non_system[self._prev_msg_count :]
            self._prev_msg_count = len(non_system)
            if new_messages:
                threading.Thread(
                    target=self._update_memory, args=(new_messages,), daemon=True
                ).start()

        print("[Mem0] Memory component stopped")

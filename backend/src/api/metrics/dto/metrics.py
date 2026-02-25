from __future__ import annotations

from pydantic import BaseModel


class SenderSnapshot(BaseModel):
    name: str
    msg_count_delta: int
    byte_count_delta: int
    last_send_time: float
    buffer_depth: int


class ReceiverSnapshot(BaseModel):
    name: str
    msg_count_delta: int
    byte_count_delta: int
    lag: int


class NodeSnapshot(BaseModel):
    name: str
    status: str
    senders: dict[str, SenderSnapshot]
    receivers: dict[str, ReceiverSnapshot]


class MetricsResponse(BaseModel):
    nodes: dict[str, NodeSnapshot]
    timestamp: float

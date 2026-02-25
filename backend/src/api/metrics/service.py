from __future__ import annotations

import time
from collections import defaultdict

from src.api.metrics.dto import (
    MetricsResponse,
    NodeSnapshot,
    ReceiverSnapshot,
    SenderSnapshot,
)
from src.core.graph import GraphManager, ReceiverKey, SenderKey


class MetricsCollector:
    def __init__(self) -> None:
        self._prev_senders: dict[SenderKey, tuple[int, int]] = {}
        self._prev_receivers: dict[ReceiverKey, tuple[int, int]] = {}

    def collect(self, manager: GraphManager) -> MetricsResponse:
        components = manager.components()
        sender_handles = manager.sender_handles()
        receiver_handles = manager.receiver_handles()

        # Group senders by node_id
        senders_by_node: dict[str, dict[str, SenderSnapshot]] = defaultdict(dict)
        new_prev_senders: dict[SenderKey, tuple[int, int]] = {}
        for (node_id, slot), sender in sender_handles.items():
            prev_msg, prev_bytes = self._prev_senders.get((node_id, slot), (0, 0))
            cur_msg = sender._msg_count
            cur_bytes = sender._byte_count
            new_prev_senders[(node_id, slot)] = (cur_msg, cur_bytes)
            senders_by_node[node_id][slot] = SenderSnapshot(
                name=slot,
                msg_count_delta=cur_msg - prev_msg,
                byte_count_delta=cur_bytes - prev_bytes,
                last_send_time=sender._last_send_time,
                buffer_depth=sender.buffer_depth,
            )
        self._prev_senders = new_prev_senders

        # Group receivers by node_id
        receivers_by_node: dict[str, dict[str, ReceiverSnapshot]] = defaultdict(dict)
        new_prev_receivers: dict[ReceiverKey, tuple[int, int]] = {}
        for (node_id, slot), receiver in receiver_handles.items():
            prev_msg, prev_bytes = self._prev_receivers.get((node_id, slot), (0, 0))
            cur_msg = receiver._msg_count
            cur_bytes = receiver._byte_count
            new_prev_receivers[(node_id, slot)] = (cur_msg, cur_bytes)
            receivers_by_node[node_id][slot] = ReceiverSnapshot(
                name=slot,
                msg_count_delta=cur_msg - prev_msg,
                byte_count_delta=cur_bytes - prev_bytes,
                lag=receiver.lag,
            )
        self._prev_receivers = new_prev_receivers

        # Build node snapshots
        nodes: dict[str, NodeSnapshot] = {}
        for node_id, comp in components.items():
            nodes[node_id] = NodeSnapshot(
                name=comp.name,
                status=comp.status.value,
                senders=senders_by_node.get(node_id, {}),
                receivers=receivers_by_node.get(node_id, {}),
            )

        return MetricsResponse(nodes=nodes, timestamp=time.time())

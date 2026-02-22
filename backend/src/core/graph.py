from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from pydantic import BaseModel

from src.core.channel import Channel, Receiver, Sender
from src.core.component import Component


SenderKey = tuple[str, str]  # (node_id, slot_name)
ReceiverKey = tuple[str, str]  # (node_id, slot_name)


class Node(BaseModel):
    type: str
    init_args: dict[str, Any]
    x: float = 0.0
    y: float = 0.0


class Edge(BaseModel):
    source_node: str
    source_slot: str
    target_node: str
    target_slot: str


class Graph(BaseModel):
    edges: list[Edge]
    nodes: dict[str, Node]


class GraphManager:
    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._components: dict[str, Component[..., Any]] = {}
        self._channel_map: dict[frozenset[SenderKey], Channel[Any]] = {}
        self._sender_handles: dict[SenderKey, Sender[Any]] = {}
        self._receiver_handles: dict[ReceiverKey, Receiver[Any]] = {}

    # --- node CRUD ---

    def add_node(self, node_type: str, init_args: dict[str, Any]) -> tuple[str, Node]:
        classes = Component.registered_subclasses()
        cls = classes.get(node_type)
        if cls is None:
            raise ValueError(f"Unknown node type: {node_type}")
        comp = cls.from_args(init_args)
        node_id = str(uuid.uuid4())
        node = Node(type=node_type, init_args=init_args)
        self._graph.nodes[node_id] = node
        self._components[node_id] = comp
        return node_id, node

    def get_node(self, node_id: str) -> Node | None:
        return self._graph.nodes.get(node_id)

    def update_node(self, node_id: str, x: float, y: float) -> Node | None:
        node = self._graph.nodes.get(node_id)
        if node is None:
            return None
        node.x = x
        node.y = y
        return node

    def delete_node(self, node_id: str) -> None:
        node = self._graph.nodes.get(node_id)
        if node is None:
            return

        comp = self._components.get(node_id)
        if comp is not None:
            comp.stop()

        # Collect connected components that need stopping
        affected: set[str] = set()
        for edge in self._graph.edges:
            if edge.source_node == node_id:
                affected.add(edge.target_node)
            if edge.target_node == node_id:
                affected.add(edge.source_node)

        self._graph.edges = [
            e
            for e in self._graph.edges
            if e.source_node != node_id and e.target_node != node_id
        ]

        for affected_id in affected:
            affected_comp = self._components.get(affected_id)
            if affected_comp is not None:
                affected_comp.stop()

        del self._graph.nodes[node_id]
        self._components.pop(node_id, None)
        self._reconcile()

    # --- edge CRUD ---

    def add_edge(self, edge: Edge) -> None:
        self._graph.edges.append(edge)
        self._reconcile()

    def delete_edge(self, edge: Edge) -> None:
        self._graph.edges.remove(edge)
        self._reconcile()

    @property
    def graph(self) -> Graph:
        return self._graph

    def component(self, node_id: str) -> Component[..., Any]:
        return self._components[node_id]

    def components(self) -> dict[str, Component[..., Any]]:
        return self._components

    def get_node_output(self, node_id: str) -> dict[str, type]:
        return type(self._components[node_id]).get_output_types()

    def get_node_input(self, node_id: str) -> dict[str, type]:
        return type(self._components[node_id]).get_input_types()

    def reset(self, graph: Graph) -> None:
        """Stop everything and replace with a new graph + components."""
        self.stop()
        self._graph = graph
        self._components.clear()
        self._channel_map.clear()
        self._sender_handles.clear()
        self._receiver_handles.clear()

    @staticmethod
    def _group(
        edges: list[tuple[SenderKey, ReceiverKey]],
    ) -> dict[frozenset[SenderKey], list[ReceiverKey]]:
        """Group receivers by identical sender set to minimize channel count."""
        recv_to_senders: dict[ReceiverKey, set[SenderKey]] = defaultdict(set)
        for sender_key, recv_key in edges:
            recv_to_senders[recv_key].add(sender_key)

        groups: dict[frozenset[SenderKey], list[ReceiverKey]] = defaultdict(list)
        for recv_key, sender_set in recv_to_senders.items():
            groups[frozenset(sender_set)].append(recv_key)

        return dict(groups)

    def _reconcile(self) -> None:
        """Recompute optimal channel layout and diff against existing."""
        edges: list[tuple[SenderKey, ReceiverKey]] = [
            ((e.source_node, e.source_slot), (e.target_node, e.target_slot))
            for e in self._graph.edges
        ]
        groups = self._group(edges)

        old_keys = set(self._channel_map.keys())
        new_keys = set(groups.keys())

        reuse = old_keys & new_keys
        create = new_keys - old_keys

        new_channel_map: dict[frozenset[SenderKey], Channel[Any]] = {}
        for key in reuse:
            new_channel_map[key] = self._channel_map[key]
        for key in create:
            new_channel_map[key] = Channel()

        self._channel_map = new_channel_map

        sender_channels: dict[SenderKey, list[Channel[Any]]] = defaultdict(list)
        for sender_set, channel in self._channel_map.items():
            for sender_key in sender_set:
                sender_channels[sender_key].append(channel)

        new_sender_handles: dict[SenderKey, Sender[Any]] = {}
        for key, channels in sender_channels.items():
            old = self._sender_handles.get(key)
            if old is not None and set(old._channels) == set(channels):
                new_sender_handles[key] = old
            else:
                new_sender_handles[key] = Sender(*channels)
        self._sender_handles = new_sender_handles

        new_receiver_handles: dict[ReceiverKey, Receiver[Any]] = {}
        for sender_set, recv_keys in groups.items():
            channel = self._channel_map[sender_set]
            for recv_key in recv_keys:
                old = self._receiver_handles.get(recv_key)
                if old is not None and old._channel is channel:
                    new_receiver_handles[recv_key] = old
                else:
                    new_receiver_handles[recv_key] = Receiver(channel)
        self._receiver_handles = new_receiver_handles

    def run(self) -> None:
        """Stop all running components, then start each with wired handles."""
        self.stop()

        for node_id in self._graph.nodes:
            comp = self._components[node_id]
            cls = type(comp)

            input_type = cls._get_type_param(0)
            output_type = cls._get_type_param(1)

            input_slots = cls.get_input_types()
            output_slots = cls.get_output_types()

            input_handles: dict[str, Receiver[Any]] = {}
            for slot in input_slots:
                key: ReceiverKey = (node_id, slot)
                if key in self._receiver_handles:
                    input_handles[slot] = self._receiver_handles[key]

            output_handles: dict[str, Sender[Any]] = {}
            for slot in output_slots:
                key: SenderKey = (node_id, slot)
                if key in self._sender_handles:
                    output_handles[slot] = self._sender_handles[key]

            inputs = self._build_tuple(input_type, input_handles)
            outputs = self._build_tuple(output_type, output_handles)

            comp.start(inputs, outputs)

    @staticmethod
    def _build_tuple(tp: type | None, handles: dict[str, Any]) -> tuple[Any, ...]:
        """Build a NamedTuple (keyword) or plain tuple (positional) from handles."""
        if tp is None or not handles:
            return ()
        if hasattr(tp, "_fields"):
            return tp(**handles)
        return tuple(handles[k] for k in sorted(handles.keys()))

    def stop(self) -> None:
        """Stop all components."""
        for comp in self._components.values():
            comp.stop()

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from src.core.channel import Channel, Receiver, Sender
from src.core.component import (
    Component,
    PrimitiveComponent,
    ThreadedComponent,
)
from src.core.log_capture import get_log_store


from src.core.utils import SenderKey, ReceiverKey


class Node(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    init_args: dict[str, Any]
    x: float = 0.0
    y: float = 0.0
    sub_graph: Graph | None = None
    senders: dict[str, Sender[Any] | None] = Field(default_factory=dict, exclude=True)
    receivers: dict[str, Receiver[Any] | None] = Field(default_factory=dict, exclude=True)


class Edge(BaseModel):
    source_node: str
    source_slot: str
    target_node: str
    target_slot: str


class Graph(BaseModel):
    edges: list[Edge]
    nodes: dict[str, Node]


# Resolve forward reference: Node.sub_graph uses Graph which is defined after Node.
Node.model_rebuild()


class GraphManager:
    def __init__(self, graph: Graph) -> None:
        self._graph = Graph(edges=[], nodes={})
        self._components: dict[str, Component[Any, Any]] = {}
        self._channel_map: dict[frozenset[SenderKey], Channel[Any]] = {}
        self.reset(graph)

    # --- node CRUD ---

    def add_primitive_node(self, type_: str, init_args: dict[str, Any]) -> tuple[str, Node]:
        classes = PrimitiveComponent.registered_subclasses()
        cls = classes.get(type_)
        if cls is None:
            raise ValueError(f"Unknown node type: {type_}")
        comp = cls.from_args(init_args)
        node = Node(type=type_, init_args=init_args)
        self._graph.nodes[node.id_] = node
        self._components[node.id_] = comp
        return node.id_, node

    def add_composite_node(
        self,
        type_: str,
        sub_graph: Graph,
        x: float = 0.0,
        y: float = 0.0,
        description: str = "",
    ) -> tuple[str, Node]:
        from src.core.component import CompositeComponent

        comp = CompositeComponent(type_, sub_graph, description=description)
        node = Node(
            type=type_,
            init_args={},
            x=x,
            y=y,
            sub_graph=sub_graph,
        )
        self._graph.nodes[node.id_] = node
        self._components[node.id_] = comp
        return node.id_, node

    def get_node(self, node_id: str) -> Node | None:
        return self._graph.nodes.get(node_id)

    def update_node(self, node_id: str, x: float, y: float) -> Node | None:
        node = self._graph.nodes.get(node_id)
        if node is None:
            return None
        node.x = x
        node.y = y
        return node

    def update_primitive_node_init_args(
        self, node_id: str, init_args: dict[str, Any]
    ) -> tuple[Node | None, bool]:
        """Replace a node's init_args and recreate its component.

        Returns (node, was_running). The caller is responsible for
        calling run() with UI channel overrides if was_running is True.
        """
        node = self._graph.nodes.get(node_id)
        if node is None:
            return None, False

        classes = PrimitiveComponent.registered_subclasses()
        cls = classes.get(node.type)
        if cls is None:
            return None, False

        was_running = any(
            c.status.value == "running" for c in self._components.values()
        )
        if was_running:
            self.stop()

        node.init_args = init_args
        self._components[node_id] = cls.from_args(init_args)
        self._reconcile()
        return node, was_running

    def delete_node(self, node_id: str) -> None:
        node = self._graph.nodes.get(node_id)
        if node is None:
            return

        comp = self._components.get(node_id)
        if comp is not None:
            comp.stop()

        # Collect downstream components that need stopping
        affected: set[str] = set()
        for edge in self._graph.edges:
            if edge.source_node == node_id:
                affected.add(edge.target_node)

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
        get_log_store().clear_node(node_id)
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

    def component(self, node_id: str) -> Component[Any, Any]:
        return self._components[node_id]

    def components(self) -> dict[str, Component[Any, Any]]:
        return self._components

    def sender_handles(self) -> dict[SenderKey, Sender[Any]]:
        return {
            (node_id, slot): sender
            for node_id, node in self._graph.nodes.items()
            for slot, sender in node.senders.items()
            if sender is not None
        }

    def receiver_handles(self) -> dict[ReceiverKey, Receiver[Any]]:
        return {
            (node_id, slot): receiver
            for node_id, node in self._graph.nodes.items()
            for slot, receiver in node.receivers.items()
            if receiver is not None
        }

    def ui_input_slots(self) -> list[ReceiverKey]:
        """All (node_id, slot) pairs for UI input slots across the graph."""
        return [
            (node_id, slot)
            for node_id, comp in self._components.items()
            for slot in comp.get_ui_input_types()
        ]

    def ui_output_slots(self) -> list[SenderKey]:
        """All (node_id, slot) pairs for UI output slots across the graph."""
        return [
            (node_id, slot)
            for node_id, comp in self._components.items()
            for slot in comp.get_ui_output_types()
        ]

    def get_node_output(self, node_id: str) -> dict[str, type]:
        return self._components[node_id].get_output_types()

    def get_node_input(self, node_id: str) -> dict[str, type]:
        return self._components[node_id].get_input_types()

    def reset(self, graph: Graph) -> None:
        """Stop everything and replace with a new graph + components."""
        self.stop()
        self._graph = graph
        self._components.clear()
        self._channel_map.clear()

        classes = PrimitiveComponent.registered_subclasses()
        for node_id, node in self._graph.nodes.items():
            if node.sub_graph is not None:
                from src.core.component import CompositeComponent

                self._components[node_id] = CompositeComponent(
                    node.type, node.sub_graph, description=""
                )
            else:
                cls = classes.get(node.type)
                if cls is not None:
                    self._components[node_id] = cls.from_args(node.init_args)

        self._reconcile()

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

    def _reconcile(
        self,
    ) -> tuple[dict[SenderKey, list[Channel[Any]]], dict[ReceiverKey, Channel[Any]]]:
        """Recompute channel topology from the current graph edges.

        Returns (sender_plan, receiver_plan) — the wiring blueprint.
        Does not create or touch Sender/Receiver handles — that happens in run().
        """
        edges: list[tuple[SenderKey, ReceiverKey]] = [
            ((e.source_node, e.source_slot), (e.target_node, e.target_slot))
            for e in self._graph.edges
        ]
        groups = self._group(edges)

        old_keys = set(self._channel_map.keys())
        new_keys = set(groups.keys())

        # step 1: create/remove channels
        new_channel_map: dict[frozenset[SenderKey], Channel[Any]] = {}
        for ckey in old_keys & new_keys:
            new_channel_map[ckey] = self._channel_map[ckey]
        for ckey in new_keys - old_keys:
            new_channel_map[ckey] = Channel()
        self._channel_map = new_channel_map

        # step 2: build the wiring plan
        sender_plan: dict[SenderKey, list[Channel[Any]]] = defaultdict(list)
        for sender_set, channel in self._channel_map.items():
            for sender_key in sender_set:
                sender_plan[sender_key].append(channel)

        receiver_plan: dict[ReceiverKey, Channel[Any]] = {}
        for sender_set, recv_keys in groups.items():
            channel = self._channel_map[sender_set]
            for recv_key in recv_keys:
                receiver_plan[recv_key] = channel

        return dict(sender_plan), receiver_plan

    def run(
        self,
        receiver_overrides: dict[ReceiverKey, Receiver[Any] | None] | None = None,
        sender_overrides: dict[SenderKey, Sender[Any] | None] | None = None,
    ) -> None:
        """Stop all running components, then start each with fresh handles.

        Optional overrides let callers (e.g. CompositeComponent) inject
        pre-built handles for specific slots.
        """
        self.stop()

        sender_plan, receiver_plan = self._reconcile()
        _recv_over = receiver_overrides or {}
        _send_over = sender_overrides or {}

        start_queue: list[tuple[str, Component[Any, Any], Any, Any]] = []
        for node_id, node in self._graph.nodes.items():
            comp = self._components[node_id]
            cls = type(comp)

            input_type = cls._get_type_param(0)
            output_type = cls._get_type_param(1)

            input_slots = comp.get_input_types()
            output_slots = comp.get_output_types()

            stop_event = comp.stop_event if isinstance(comp, ThreadedComponent) else threading.Event()

            # Create fresh handles from plan, store on node.
            # Overrides take priority over the plan.
            for slot in input_slots:
                if (node_id, slot) in _recv_over:
                    node.receivers[slot] = _recv_over[(node_id, slot)]
                elif (node_id, slot) in receiver_plan:
                    node.receivers[slot] = Receiver(receiver_plan[(node_id, slot)], stop_event)
                else:
                    node.receivers[slot] = None

            for slot in output_slots:
                if (node_id, slot) in _send_over:
                    node.senders[slot] = _send_over[(node_id, slot)] or Sender()
                elif (node_id, slot) in sender_plan:
                    node.senders[slot] = Sender(*sender_plan[(node_id, slot)])
                else:
                    # Unconnected output: no-op sender (sends are discarded)
                    node.senders[slot] = Sender()

            built_inputs = self._build_tuple(input_type, dict(node.receivers))
            built_outputs = self._build_tuple(output_type, dict(node.senders))

            start_queue.append((node_id, comp, built_inputs, built_outputs))

        # Call setup() on all components AFTER all receivers are wired,
        # so initial data lands behind every cursor regardless of node order.
        for _, comp, _, built_outputs in start_queue:
            comp.setup(built_outputs)

        # Start all components after all emits are done
        for node_id, comp, inputs, outputs in start_queue:
            comp.start(inputs, outputs)
            if isinstance(comp, ThreadedComponent):
                ident = comp.get_ident()
                if ident is None:
                    for _ in range(10):
                        time.sleep(0.005)
                        ident = comp.get_ident()
                        if ident is not None:
                            break
                if ident is not None:
                    get_log_store().register_thread(node_id=node_id, ident=ident)

    @staticmethod
    def _build_tuple(tp: type | None, handles: dict[str, Any]) -> tuple[Any, ...]:
        """Build a NamedTuple (keyword) or plain tuple (positional) from handles."""
        if tp is None or not handles:
            return ()
        if hasattr(tp, "_fields"):
            return tp(**handles)
        # Composite or plain tuple: preserve insertion order of handles dict
        return tuple(handles.values())

    def stop(self) -> None:
        """Stop all components and await their threads."""
        for node in self._graph.nodes.values():
            for sender in node.senders.values():
                if sender is not None:
                    sender._stopped = True
        for comp in self._components.values():
            comp.stop()
        for comp in self._components.values():
            if isinstance(comp, ThreadedComponent):
                comp.join(timeout=5.0)

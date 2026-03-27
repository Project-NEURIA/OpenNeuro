from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Any, get_args, get_origin

from pydantic import BaseModel, Field

from src.core.channel import Channel, Receiver, Sender
from src.core.component import (
    Component,
    EmitOnStart,
    PrimitiveComponent,
    ThreadedComponent,
)
from src.core.log_capture import get_log_store


SenderKey = tuple[str, str]  # (node_id, slot_name)
ReceiverKey = tuple[str, str]  # (node_id, slot_name)


class Node(BaseModel):
    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    init_args: dict[str, Any]
    x: float = 0.0
    y: float = 0.0
    sub_graph: Graph | None = None


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
        self._sender_handles: dict[SenderKey, Sender[Any]] = {}
        self._receiver_handles: dict[ReceiverKey, Receiver[Any]] = {}
        # UI channels: keyed by (node_id, slot_name)
        self._ui_channels: dict[tuple[str, str], Channel[Any]] = {}
        self._ui_senders: dict[tuple[str, str], Sender[Any]] = {}
        self._ui_receivers: dict[tuple[str, str], Receiver[Any]] = {}
        self._ui_version = 0
        self._ui_changed = asyncio.Event()
        self.reset(graph)

    # --- node CRUD ---

    def add_node(self, node_type: str, init_args: dict[str, Any]) -> tuple[str, Node]:
        classes = PrimitiveComponent.registered_subclasses()
        cls = classes.get(node_type)
        if cls is None:
            raise ValueError(f"Unknown node type: {node_type}")
        comp = cls.from_args(init_args)
        node = Node(type=node_type, init_args=init_args)
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

    def update_node_init_args(
        self, node_id: str, init_args: dict[str, Any]
    ) -> Node | None:
        node = self._graph.nodes.get(node_id)
        if node is None:
            return None

        classes = PrimitiveComponent.registered_subclasses()
        cls = classes.get(node.type)
        if cls is None:
            return None

        was_running = any(
            c.status.value == "running" for c in self._components.values()
        )
        if was_running:
            self.stop()

        node.init_args = init_args
        self._components[node_id] = cls.from_args(init_args)
        self._reconcile()

        if was_running:
            self.run()
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
        return self._sender_handles

    def receiver_handles(self) -> dict[ReceiverKey, Receiver[Any]]:
        return self._receiver_handles

    def ui_senders(self) -> dict[tuple[str, str], Sender[Any]]:
        """Server-side senders that push data into component UIReceiver slots."""
        return self._ui_senders

    def ui_receivers(self) -> dict[tuple[str, str], Receiver[Any]]:
        """Server-side receivers that read from component UISender slots."""
        return self._ui_receivers

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
        self._sender_handles.clear()
        self._receiver_handles.clear()
        self._ui_channels.clear()
        self._ui_senders.clear()
        self._ui_receivers.clear()

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
        for ckey in reuse:
            new_channel_map[ckey] = self._channel_map[ckey]
        for ckey in create:
            new_channel_map[ckey] = Channel()

        self._channel_map = new_channel_map

        sender_channels: dict[SenderKey, list[Channel[Any]]] = defaultdict(list)
        for sender_set, channel in self._channel_map.items():
            for sender_key in sender_set:
                sender_channels[sender_key].append(channel)

        new_sender_handles: dict[SenderKey, Sender[Any]] = {}
        for skey, channels in sender_channels.items():
            old_sender = self._sender_handles.get(skey)
            if old_sender is not None and set(old_sender._channels) == set(channels):
                new_sender_handles[skey] = old_sender
            else:
                new_sender_handles[skey] = Sender(*channels)

        # Ensure every declared output slot has a sender handle, even when no edges
        # are connected. This keeps per-output metrics (e.g. last_send_time) visible
        # for standalone sources.
        for node_id, comp in self._components.items():
            for slot in comp.get_output_types():
                output_key: SenderKey = (node_id, slot)
                if output_key in new_sender_handles:
                    continue
                old_sender = self._sender_handles.get(output_key)
                if old_sender is not None and len(old_sender._channels) == 0:
                    new_sender_handles[output_key] = old_sender
                else:
                    new_sender_handles[output_key] = Sender()
        self._sender_handles = new_sender_handles

        new_receiver_handles: dict[ReceiverKey, Receiver[Any]] = {}
        for sender_set, recv_keys in groups.items():
            channel = self._channel_map[sender_set]
            for recv_key in recv_keys:
                old_recv: Receiver[Any] | None = self._receiver_handles.get(recv_key)  # type: ignore[assignment]
                if old_recv is not None and old_recv._channel is channel:
                    new_receiver_handles[recv_key] = old_recv
                else:
                    new_receiver_handles[recv_key] = Receiver(channel)
        self._receiver_handles = new_receiver_handles

    def run(self) -> None:
        """Stop all running components, then start each with wired handles."""
        self.stop()
        for sender in self._sender_handles.values():
            sender._stopped = False
        self._ui_channels.clear()
        self._ui_senders.clear()
        self._ui_receivers.clear()
        start_queue: list[tuple[str, Component[Any, Any], Any, Any]] = []
        for node_id in self._graph.nodes:
            comp = self._components[node_id]
            cls = type(comp)

            input_type = cls._get_type_param(0)
            output_type = cls._get_type_param(1)

            # Use instance calls so CompositeComponent overrides take effect
            input_slots = comp.get_input_types()
            output_slots = comp.get_output_types()
            ui_input_slots = comp.get_ui_input_types()
            ui_output_slots = comp.get_ui_output_types()

            from src.core.component import CompositeComponent

            is_composite = isinstance(comp, CompositeComponent)

            input_handles: dict[str, Receiver[Any] | None] = {}
            for slot, slot_type in input_slots.items():
                rkey: ReceiverKey = (node_id, slot)
                if rkey in self._receiver_handles:
                    input_handles[slot] = self._receiver_handles[rkey]
                elif type(None) in get_args(slot_type):
                    # Optional inputs with no edge get None so NamedTuple
                    # construction doesn't fail from a missing field.
                    input_handles[slot] = None

            output_handles: dict[str, Sender[Any]] = {}
            for slot in output_slots:
                skey: SenderKey = (node_id, slot)
                if skey in self._sender_handles:
                    output_handles[slot] = self._sender_handles[skey]
                else:
                    # Unconnected output: no-op Sender (sends are discarded)
                    output_handles[slot] = Sender()

            # Wire UI input channels (frontend -> component)
            for slot, slot_type in ui_input_slots.items():
                ch: Channel[Any] = Channel()
                self._ui_channels[(node_id, slot)] = ch
                # Component gets a UIReceiver to read from
                origin = get_origin(slot_type) or slot_type
                input_handles[slot] = origin(ch)
                # Server keeps a Sender to push data in
                self._ui_senders[(node_id, slot)] = Sender(ch)

            # Wire UI output channels (component -> frontend)
            for slot, slot_type in ui_output_slots.items():
                ch = Channel()
                self._ui_channels[(node_id, slot)] = ch
                # Component gets a UISender to write to
                origin = get_origin(slot_type) or slot_type
                output_handles[slot] = origin(ch)
                # Server keeps a Receiver to read from
                self._ui_receivers[(node_id, slot)] = Receiver(ch)

            # Wire all receivers (registers cursors before EmitOnStart)
            if isinstance(comp, ThreadedComponent):
                for handle in input_handles.values():
                    if isinstance(handle, Receiver):
                        handle._wire(comp.stop_event)

            if is_composite:
                built_inputs: Any = input_handles
                built_outputs: Any = output_handles
            else:
                built_inputs = self._build_tuple(input_type, input_handles)
                built_outputs = self._build_tuple(output_type, output_handles)

            if isinstance(comp, EmitOnStart):
                comp.emit(built_outputs)
            start_queue.append((node_id, comp, built_inputs, built_outputs))

        # Start all components after all emits are done
        for node_id, comp, inputs, outputs in start_queue:
            comp.start(inputs, outputs)
            if isinstance(comp, ThreadedComponent):
                ident = comp.get_ident()
                if ident is None:
                    # Thread ident may lag briefly after start().
                    for _ in range(10):
                        time.sleep(0.005)
                        ident = comp.get_ident()
                        if ident is not None:
                            break
                if ident is not None:
                    get_log_store().register_thread(node_id=node_id, ident=ident)

        # Notify WS listeners that UI channels are ready.
        # Save old event, replace with fresh one, *then* wake waiters.
        # This avoids a race where a coroutine calling wait() between
        # set() and reassignment would block on the now-orphaned event.
        self._ui_version += 1
        old_event = self._ui_changed
        self._ui_changed = asyncio.Event()
        old_event.set()

    @staticmethod
    def _build_tuple(tp: type | None, handles: dict[str, Any]) -> tuple[Any, ...]:
        """Build a NamedTuple (keyword) or plain tuple (positional) from handles."""
        if tp is None or not handles:
            return ()
        if hasattr(tp, "_fields"):
            return tp(**handles)
        return tuple(handles[k] for k in sorted(handles.keys()))

    def stop(self) -> None:
        """Stop all components and senders."""
        for sender in self._sender_handles.values():
            sender._stopped = True
        for sender in self._ui_senders.values():
            sender._stopped = True
        for comp in self._components.values():
            comp.stop()

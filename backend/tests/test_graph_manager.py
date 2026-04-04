"""Unit tests for GraphManager (backend/src/core/graph.py).

Module Under Test — GraphManager
================================
GraphManager is the runtime orchestrator for a directed dataflow graph of
audio/ML pipeline components.  Its responsibilities are:

  Responsibility            | Interface method(s)
  ------------------------- | -------------------------------------------
  Node lifecycle            | add_primitive_node, delete_node, get_node, update_node
  Edge lifecycle            | add_edge, delete_edge
  Hot-reload of init args   | update_primitive_node_init_args
  Channel reconciliation    | _reconcile (private), _group (static)
  Pipeline start / stop     | run, stop, reset
  Handle construction       | _build_tuple (static)
  UI channel management     | ui_senders, ui_receivers (properties)

Contract:
  - Every node maps to exactly one Component instance.
  - Edges declare (source_node, source_slot) → (target_node, target_slot)
    connections.  The channel layout is always minimal: the fewest channels
    such that every receiver's attached channel carries data from exactly the
    set of senders specified by the graph's edges — no more (which would
    introduce unwanted input) and no fewer (which would silently drop
    connections).  Channels from the previous layout are reused where possible
    to preserve in-flight data.
  - Deleting a node cascades: all edges touching that node are removed,
    affected downstream components are stopped, and a full reconcile runs.
  - run() wires Sender/Receiver handles into each Component via NamedTuples,
    then starts every component.  stop() is idempotent.

Edge cases tested below:
  - Unknown node type (should raise ValueError)
  - Deleting / updating a non-existent node (should be a safe no-op)
  - Channel reuse across reconciliation rounds
  - Grouping receivers that share the same sender set
  - NamedTuple vs empty-tuple construction in _build_tuple
  - Full run/stop lifecycle with real (stub) threaded components
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent, ThreadedComponent, Status
from src.core.graph import Edge, Graph, GraphManager


# ---------------------------------------------------------------------------
# Stub component — lightweight stand-in so we never import heavy deps
# ---------------------------------------------------------------------------


class StubInputs(NamedTuple):
    data: Receiver[str]


class StubOutputs(NamedTuple):
    data: Sender[str]


class StubComponent(ThreadedComponent[StubInputs, StubOutputs]):
    """Minimal component for testing.  Its run() loop just blocks until stop."""

    def run(self, inputs: StubInputs, outputs: StubOutputs) -> None:
        while not self.stop_event.is_set():
            self.stop_event.wait(0.05)


def _stub_registry() -> dict[str, type[PrimitiveComponent[Any, Any]]]:
    """Return a registry containing only StubComponent."""
    return {"StubComponent": StubComponent}


# Patch the registry for all tests so we never trigger real component imports.
@pytest.fixture(autouse=True)
def _patch_registry():
    with patch.object(
        PrimitiveComponent, "registered_subclasses", return_value=_stub_registry()
    ):
        yield


def _empty_graph() -> Graph:
    return Graph(edges=[], nodes={})


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


class TestAddNode:
    """Test 1: Adding a valid node creates both a graph entry and a component.

    Why: This is the fundamental happy-path — every other operation depends on
    nodes being correctly registered in both _graph.nodes and _components.
    Edge case: the returned node must have the correct type field, and the
    manager's internal dicts must stay in sync.
    """

    def test_add_primitive_node(self):
        gm = GraphManager(_empty_graph())
        node_id, node = gm.add_primitive_node("StubComponent", {})

        assert node_id in gm.graph.nodes
        assert gm.graph.nodes[node_id].type == "StubComponent"
        assert node_id in gm.components()
        assert isinstance(gm.components()[node_id], StubComponent)


class TestAddAndDeleteEdge:
    """Test 2: Adding an edge wires a channel; deleting it unwires.

    Why: Edges drive channel creation through _reconcile().  After adding an
    edge the sender/receiver handle maps must contain entries for the connected
    slots.  After deleting, those handles must be cleaned up.
    Edge case: verifies that reconcile is idempotent — the graph returns to a
    clean state after the edge is removed.
    """

    def test_add_and_delete_edge(self):
        gm = GraphManager(_empty_graph())
        id_a, _ = gm.add_primitive_node("StubComponent", {})
        id_b, _ = gm.add_primitive_node("StubComponent", {})

        edge = Edge(
            source_node=id_a,
            source_slot="data",
            target_node=id_b,
            target_slot="data",
        )
        gm.add_edge(edge)
        gm.run()

        # After run: receiver handle exists for the target slot
        assert (id_b, "data") in gm.receiver_handles()
        # Sender handle for source slot should be wired to at least one channel
        sender = gm.sender_handles().get((id_a, "data"))
        assert sender is not None
        assert len(sender._channels) >= 1

        gm.stop()
        gm.delete_edge(edge)
        gm.run()

        # After deleting and re-running: sender has no channels (unconnected)
        sender = gm.sender_handles().get((id_a, "data"))
        assert sender is not None
        assert len(sender._channels) == 0


class TestDeleteNodeRemovesEdges:
    """Test 3: Deleting a node cascades to remove all its connected edges.

    Why: If edges referencing a deleted node linger, _reconcile() would create
    orphaned channels or KeyError on missing component lookups.
    Edge case: edges in *both* directions (outgoing and incoming) must be
    removed, and the affected neighbour components must be stopped.
    """

    def test_delete_node_removes_edges(self):
        gm = GraphManager(_empty_graph())
        id_a, _ = gm.add_primitive_node("StubComponent", {})
        id_b, _ = gm.add_primitive_node("StubComponent", {})
        id_c, _ = gm.add_primitive_node("StubComponent", {})

        gm.add_edge(
            Edge(
                source_node=id_a,
                source_slot="data",
                target_node=id_b,
                target_slot="data",
            )
        )
        gm.add_edge(
            Edge(
                source_node=id_b,
                source_slot="data",
                target_node=id_c,
                target_slot="data",
            )
        )

        assert len(gm.graph.edges) == 2

        # Delete the middle node — both edges should be removed
        gm.delete_node(id_b)

        assert len(gm.graph.edges) == 0
        assert id_b not in gm.graph.nodes
        assert id_b not in gm.components()


class TestReconcileReusesChannels:
    """Test 4: Adding a second edge does not destroy the first edge's channel.

    Why: Channel reuse is critical for correctness — if an existing channel is
    unnecessarily recreated, any in-flight data is lost.  _reconcile() must
    diff old vs new channel keys and keep unchanged ones.
    Edge case: the channel object identity (id()) must be the same before and
    after adding an unrelated edge.
    """

    def test_reconcile_reuses_channels(self):
        gm = GraphManager(_empty_graph())
        id_a, _ = gm.add_primitive_node("StubComponent", {})
        id_b, _ = gm.add_primitive_node("StubComponent", {})
        id_c, _ = gm.add_primitive_node("StubComponent", {})

        edge_ab = Edge(
            source_node=id_a,
            source_slot="data",
            target_node=id_b,
            target_slot="data",
        )
        gm.add_edge(edge_ab)

        # Capture the channel used for A→B
        channel_key = frozenset({(id_a, "data")})
        channel_before = gm._channel_map.get(channel_key)
        assert channel_before is not None

        # Add unrelated edge B→C — A→B channel should survive
        gm.add_edge(
            Edge(
                source_node=id_b,
                source_slot="data",
                target_node=id_c,
                target_slot="data",
            )
        )

        channel_after = gm._channel_map.get(channel_key)
        assert channel_after is channel_before  # same object, not recreated


class TestGroupMergesIdenticalSenders:
    """Test 5: _group() merges receivers that share the same sender set.

    Why: The grouping strategy is the core of channel minimisation.  Two
    receivers fed by exactly the same set of senders should share one channel.
    Edge case: order of edges should not matter — the frozenset key makes
    grouping order-independent.
    """

    def test_group_merges(self):
        edges = [
            (("A", "out"), ("B", "in")),
            (("A", "out"), ("C", "in")),
        ]
        groups = GraphManager._group(edges)

        # B and C both receive from {("A","out")} — one group
        key = frozenset({("A", "out")})
        assert key in groups
        assert set(groups[key]) == {("B", "in"), ("C", "in")}

    def test_group_separates_different_senders(self):
        """Receivers with different sender sets must NOT share a channel."""
        edges = [
            (("A", "out"), ("C", "in")),
            (("B", "out"), ("D", "in")),
        ]
        groups = GraphManager._group(edges)

        assert len(groups) == 2
        assert frozenset({("A", "out")}) in groups
        assert frozenset({("B", "out")}) in groups


class TestUpdateNodePosition:
    """Test 6: update_node() correctly sets x/y coordinates.

    Why: Position is used by the frontend to render the graph.  A simple
    setter, but must return the updated Node (not None) for existing nodes.
    Edge case: verifying the return value is the *same* Node object that
    lives in the graph (not a stale copy).
    """

    def test_update_node_position(self):
        gm = GraphManager(_empty_graph())
        node_id, _ = gm.add_primitive_node("StubComponent", {})

        result = gm.update_node(node_id, x=42.0, y=99.0)

        assert result is not None
        assert result.x == 42.0
        assert result.y == 99.0
        assert gm.graph.nodes[node_id].x == 42.0


class TestBuildTuple:
    """Tests 7-8: _build_tuple constructs NamedTuples or returns empty tuple.

    Why: _build_tuple is the bridge between the handle dicts and the typed
    I/O tuples that components receive.  Incorrect construction causes
    TypeErrors or missing handles at runtime.
    Edge cases:
      - NamedTuple with keyword construction (field-name matching)
      - None type → empty tuple (source/sink with no inputs/outputs)
      - Empty handles dict → empty tuple
    """

    def test_build_namedtuple(self):
        class MyTuple(NamedTuple):
            a: int
            b: str

        result = GraphManager._build_tuple(MyTuple, {"a": 1, "b": "hello"})
        assert isinstance(result, MyTuple)
        assert result.a == 1
        assert result.b == "hello"

    def test_build_empty_for_none_type(self):
        result = GraphManager._build_tuple(None, {})
        assert result == ()

    def test_build_empty_for_empty_handles(self):
        class MyTuple(NamedTuple):
            x: int

        result = GraphManager._build_tuple(MyTuple, {})
        assert result == ()


class TestRunAndStopLifecycle:
    """Test 9: run() starts components; stop() stops them.

    Why: This is the integration-level happy path.  If run/stop don't
    correctly wire handles and manage component status, the whole pipeline
    is broken.
    Edge case: calling stop() twice must be idempotent (no crash).
    """

    def test_run_and_stop(self):
        gm = GraphManager(_empty_graph())
        id_a, _ = gm.add_primitive_node("StubComponent", {})
        id_b, _ = gm.add_primitive_node("StubComponent", {})
        gm.add_edge(
            Edge(
                source_node=id_a,
                source_slot="data",
                target_node=id_b,
                target_slot="data",
            )
        )

        gm.run()
        # Give threads a moment to start
        time.sleep(0.1)

        for comp in gm.components().values():
            assert comp.status in (Status.RUNNING, Status.SETUP)

        gm.stop()
        time.sleep(0.1)

        # Idempotent — second stop should not raise
        gm.stop()


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


class TestAddNodeUnknownType:
    """Test 10: Adding a node with an unknown type raises ValueError.

    Why: The registry lookup is the first validation gate.  A clear error
    message here prevents confusing downstream KeyErrors.
    Edge case: typos in node type names must fail loudly, not create a
    partially-initialised node.
    """

    def test_add_primitive_node_unknown_type(self):
        gm = GraphManager(_empty_graph())
        with pytest.raises(ValueError, match="Unknown node type"):
            gm.add_primitive_node("NonExistentComponent", {})

        # Graph must remain unchanged after the failed add
        assert len(gm.graph.nodes) == 0


class TestDeleteNonexistentNode:
    """Test 11: Deleting a node that doesn't exist is a silent no-op.

    Why: In a concurrent UI, a delete request might arrive for a node that
    was already removed.  This must not crash.
    Edge case: no KeyError, no side effects on existing nodes.
    """

    def test_delete_nonexistent_node(self):
        gm = GraphManager(_empty_graph())
        id_a, _ = gm.add_primitive_node("StubComponent", {})

        # Should not raise
        gm.delete_node("does-not-exist")

        # Existing node untouched
        assert id_a in gm.graph.nodes


class TestUpdateNonexistentNode:
    """Test 12: Updating a non-existent node returns None.

    Why: Same rationale as delete — the API must handle stale references
    gracefully.  Returning None lets callers distinguish "not found" from
    "updated".
    Edge case: update_node and update_primitive_node_init_args both return None.
    """

    def test_update_nonexistent_node_position(self):
        gm = GraphManager(_empty_graph())
        result = gm.update_node("ghost", x=1.0, y=2.0)
        assert result is None

    def test_update_nonexistent_node_init_args(self):
        gm = GraphManager(_empty_graph())
        node, was_running = gm.update_primitive_node_init_args("ghost", {"key": "value"})
        assert node is None
        assert was_running is False

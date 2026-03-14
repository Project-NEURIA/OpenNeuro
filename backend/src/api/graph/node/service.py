from __future__ import annotations

from typing import Any

from src.api.graph.node.dto import NodeUpdateRequest
from src.core.graph import Edge, Graph, GraphManager, Node


def list_nodes(manager: GraphManager) -> dict[str, Node]:
    return manager.graph.nodes


def get_node(manager: GraphManager, node_id: str) -> Node | None:
    return manager.get_node(node_id)


def create_node(
    manager: GraphManager, node_type: str, init_args: dict[str, Any]
) -> tuple[str, Node]:
    return manager.add_node(node_type, init_args)


def update_node(
    manager: GraphManager, node_id: str, req: NodeUpdateRequest
) -> Node | None:
    return manager.update_node(node_id, x=req.x, y=req.y)


def delete_node(manager: GraphManager, node_id: str) -> None:
    manager.delete_node(node_id)


def create_subgraph(
    manager: GraphManager, node_ids: list[str], name: str = "Subgraph"
) -> tuple[str, Node]:
    """Group selected nodes into a composite node.

    1. Extract selected nodes + internal edges into a sub_graph
    2. Create a composite node with derived external ports
    3. Rewire external edges to point to/from the composite
    4. Delete original nodes from outer graph
    """
    selected = set(node_ids)
    graph = manager.graph

    # Validate all node IDs exist
    for nid in node_ids:
        if nid not in graph.nodes:
            raise ValueError(f"Node not found: {nid}")

    # Partition edges
    internal_edges: list[Edge] = []
    # Edges crossing the boundary: one end inside, one outside
    inbound_edges: list[Edge] = []  # outside → inside
    outbound_edges: list[Edge] = []  # inside → outside
    remaining_edges: list[Edge] = []  # outside → outside (untouched)

    for edge in graph.edges:
        src_in = edge.source_node in selected
        tgt_in = edge.target_node in selected
        if src_in and tgt_in:
            internal_edges.append(edge)
        elif not src_in and tgt_in:
            inbound_edges.append(edge)
        elif src_in and not tgt_in:
            outbound_edges.append(edge)
        else:
            remaining_edges.append(edge)

    # Build sub_graph
    sub_nodes = {nid: graph.nodes[nid] for nid in node_ids}
    sub_graph = Graph(nodes=sub_nodes, edges=internal_edges)

    # Compute position: average of selected nodes
    xs = [n.x for n in sub_nodes.values()]
    ys = [n.y for n in sub_nodes.values()]
    cx = sum(xs) / len(xs) if xs else 0.0
    cy = sum(ys) / len(ys) if ys else 0.0

    # Create composite node
    comp_id, comp_node = manager.add_composite_node(sub_graph, x=cx, y=cy, label=name)

    # Rewire boundary edges to point to/from the composite node
    rewired_edges: list[Edge] = []
    for edge in inbound_edges:
        # outside → inside becomes outside → composite
        rewired_edges.append(
            Edge(
                source_node=edge.source_node,
                source_slot=edge.source_slot,
                target_node=comp_id,
                target_slot=f"{edge.target_node}.{edge.target_slot}",
            )
        )
    for edge in outbound_edges:
        # inside → outside becomes composite → outside
        rewired_edges.append(
            Edge(
                source_node=comp_id,
                source_slot=f"{edge.source_node}.{edge.source_slot}",
                target_node=edge.target_node,
                target_slot=edge.target_slot,
            )
        )

    # Remove original nodes (stop components, remove from graph)
    for nid in node_ids:
        comp = manager.components().get(nid)
        if comp is not None:
            comp.stop()
        graph.nodes.pop(nid, None)
        manager.components().pop(nid, None)

    # Replace edges: keep remaining + add rewired
    graph.edges = remaining_edges + rewired_edges

    manager._reconcile()

    return comp_id, comp_node


def ungroup(manager: GraphManager, node_id: str) -> None:
    """Reverse a subgraph grouping: restore inner nodes and rewire edges."""
    node = manager.get_node(node_id)
    if node is None:
        raise ValueError(f"Node not found: {node_id}")
    if node.sub_graph is None:
        raise ValueError(f"Node {node_id} is not a composite")

    graph = manager.graph
    sub = node.sub_graph

    # Stop the composite component
    comp = manager.components().get(node_id)
    if comp is not None:
        comp.stop()

    # Restore inner nodes to outer graph
    for inner_id, inner_node in sub.nodes.items():
        graph.nodes[inner_id] = inner_node

    # Restore inner edges
    for edge in sub.edges:
        graph.edges.append(edge)

    # Rewire edges that connect to/from the composite back to inner nodes
    new_edges: list[Edge] = []
    for edge in graph.edges:
        if edge.target_node == node_id:
            # composite port name is "{inner_node}.{slot}"
            parts = edge.target_slot.split(".", 1)
            if len(parts) == 2:
                new_edges.append(
                    Edge(
                        source_node=edge.source_node,
                        source_slot=edge.source_slot,
                        target_node=parts[0],
                        target_slot=parts[1],
                    )
                )
            else:
                new_edges.append(edge)
        elif edge.source_node == node_id:
            parts = edge.source_slot.split(".", 1)
            if len(parts) == 2:
                new_edges.append(
                    Edge(
                        source_node=parts[0],
                        source_slot=parts[1],
                        target_node=edge.target_node,
                        target_slot=edge.target_slot,
                    )
                )
            else:
                new_edges.append(edge)
        else:
            new_edges.append(edge)
    graph.edges = new_edges

    # Remove composite node
    graph.nodes.pop(node_id, None)
    manager.components().pop(node_id, None)

    # Re-instantiate the inner components
    from src.core.component import Component

    classes = Component.registered_subclasses()
    for inner_id, inner_node in sub.nodes.items():
        cls = classes.get(inner_node.type)
        if cls is not None:
            manager.components()[inner_id] = cls.from_args(inner_node.init_args)

    manager._reconcile()

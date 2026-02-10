import { useCallback, useEffect, useRef } from "react";
import {
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  addEdge,
  useReactFlow,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type Node,
  type Edge,
} from "@xyflow/react";
import { PipelineCanvas } from "@/components/pipeline/PipelineCanvas";
import { NodeSidebar } from "@/components/pipeline/NodeSidebar";
import { MetricsOverlay } from "@/components/pipeline/MetricsOverlay";
import { usePipelineData, type PipelineNodeData } from "@/hooks/usePipelineData";
import { useComponents } from "@/hooks/useComponents";
import { layoutNodes } from "@/lib/layout";
import {
  fetchNodes as apiFetchNodes,
  fetchEdges as apiFetchEdges,
  createNode as apiCreateNode,
  deleteNode as apiDeleteNode,
  createEdge as apiCreateEdge,
  deleteEdge as apiDeleteEdge,
} from "@/lib/api";
import type { ComponentInfo } from "@/lib/types";

function parseSlot(handleId: string | null | undefined): number {
  if (!handleId) return 0;
  const parts = handleId.split("-");
  return parseInt(parts[1] ?? "0", 10);
}

function deleteEdgeFromReactFlow(edge: Edge) {
  const sourceSlot = parseSlot(edge.sourceHandle);
  const targetSlot = parseSlot(edge.targetHandle);
  apiDeleteEdge(edge.source, sourceSlot, edge.target, targetSlot).catch(console.error);
}

function AppInner() {
  const components = useComponents();
  const {
    connected,
    metrics,
    componentMap,
  } = usePipelineData(components);

  const [nodes, setNodes, onNodesChangeRaw] = useNodesState<Node>([] as Node[]);
  const [edges, setEdges, onEdgesChangeRaw] = useEdgesState<Edge>([] as Edge[]);
  const initialized = useRef(false);
  const { screenToFlowPosition } = useReactFlow();

  // Initialize: fetch existing graph from backend
  useEffect(() => {
    if (initialized.current || components.length === 0) return;
    initialized.current = true;

    (async () => {
      try {
        const [backendNodes, backendEdges] = await Promise.all([
          apiFetchNodes(),
          apiFetchEdges(),
        ]);

        const nodeSpecs = backendNodes.map((n) => ({ id: n.id, type: n.type }));
        const edgeSpecs = backendEdges.map((e) => ({
          source: e.source_node,
          target: e.target_node,
        }));
        const positions = layoutNodes(nodeSpecs, edgeSpecs);
        const posMap = new Map(positions.map((p) => [p.id, p]));

        setNodes(
          backendNodes.map((n) => {
            const pos = posMap.get(n.id) ?? { x: 0, y: 0 };
            const info = componentMap[n.type];

            return {
              id: n.id,
              type: "pipeline",
              position: { x: pos.x, y: pos.y },
              data: {
                label: n.id,
                category: info?.category ?? "conduit",
                inputs: info?.inputs ?? [],
                outputs: info?.outputs ?? [],
                status: n.status,
                nodeMetrics: null,
              } satisfies PipelineNodeData,
            };
          })
        );

        setEdges(
          backendEdges.map((e) => ({
            id: `${e.source_node}:${e.source_slot}->${e.target_node}:${e.target_slot}`,
            source: e.source_node,
            sourceHandle: `out-${e.source_slot}`,
            target: e.target_node,
            targetHandle: `in-${e.target_slot}`,
            type: "pipeline",
            data: { topicName: "", msgPerSec: 0 },
          }))
        );
      } catch (err) {
        console.error("[pipeline] Init failed:", err);
      }
    })();
  }, [components, componentMap, setNodes, setEdges]);

  // Update node data with metrics (preserve positions)
  useEffect(() => {
    if (!initialized.current || !metrics) return;
    setNodes((prev) =>
      prev.map((n) => {
        const nodeMetrics = metrics.nodes[n.id] ?? null;
        const status = nodeMetrics?.status ?? (n.data as PipelineNodeData).status;

        return {
          ...n,
          data: {
            ...(n.data as PipelineNodeData),
            status,
            nodeMetrics,
          },
        };
      })
    );
  }, [metrics, setNodes]);

  // Wrap node changes — detect removals and call backend
  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      const removals = changes.filter((c) => c.type === "remove");
      onNodesChangeRaw(changes);

      for (const r of removals) {
        if (r.type === "remove") {
          setEdges((currentEdges) => {
            for (const e of currentEdges) {
              if (e.source === r.id || e.target === r.id) {
                deleteEdgeFromReactFlow(e);
              }
            }
            return currentEdges.filter(
              (e) => e.source !== r.id && e.target !== r.id
            );
          });
          apiDeleteNode(r.id).catch(console.error);
        }
      }
    },
    [onNodesChangeRaw, setEdges]
  );

  // Wrap edge changes — detect removals and call backend
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      setEdges((currentEdges) => {
        for (const c of changes) {
          if (c.type === "remove") {
            const edge = currentEdges.find((e) => e.id === c.id);
            if (edge) {
              deleteEdgeFromReactFlow(edge);
            }
          }
        }
        return currentEdges;
      });
      onEdgesChangeRaw(changes);
    },
    [onEdgesChangeRaw, setEdges]
  );

  // Handle new edge connections
  const onConnect: OnConnect = useCallback(
    (connection) => {
      setEdges((eds) =>
        addEdge(
          { ...connection, type: "pipeline", data: { topicName: "", msgPerSec: 0 } },
          eds
        )
      );
      if (connection.source && connection.target) {
        const sourceSlot = parseSlot(connection.sourceHandle);
        const targetSlot = parseSlot(connection.targetHandle);
        apiCreateEdge(connection.source, sourceSlot, connection.target, targetSlot).catch(console.error);
      }
    },
    [setEdges]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData("application/pipeline-node");
      if (!raw) return;

      const item = JSON.parse(raw) as ComponentInfo;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });

      const newNode: Node<PipelineNodeData> = {
        id: item.name,
        type: "pipeline",
        position,
        data: {
          label: item.name,
          category: item.category,
          inputs: item.inputs,
          outputs: item.outputs,
          status: "startup",
          nodeMetrics: null,
        },
      };

      setNodes((nds) => {
        if (nds.some((n) => n.id === newNode.id)) return nds;
        return [...nds, newNode];
      });

      apiCreateNode(item.name, item.name).catch(console.error);
    },
    [screenToFlowPosition, setNodes]
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <NodeSidebar components={components} />
      <div className="relative flex-1">
        <MetricsOverlay connected={connected} metrics={metrics} />
        <PipelineCanvas
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onDrop={onDrop}
          onDragOver={onDragOver}
        />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <AppInner />
    </ReactFlowProvider>
  );
}

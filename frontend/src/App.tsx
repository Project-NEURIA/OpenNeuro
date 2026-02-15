import { useCallback, useEffect, useRef, useState } from "react";
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
import { MetricsDashboard } from "@/components/metrics/MetricsDashboard";
import { usePipelineData, type PipelineNodeData } from "@/hooks/usePipelineData";
import { useComponents } from "@/hooks/useComponents";
import { useMetricsHistory } from "@/hooks/useMetricsHistory";
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

function parseSlot(handleId: string | null | undefined): string {
  if (!handleId) return "";
  const parts = handleId.split("-");
  return parts.slice(1).join("-");
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

  const [metricsOpen, setMetricsOpen] = useState(false);
  const history = useMetricsHistory(metrics);

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
                label: n.type,
                category: info?.category ?? "conduit",
                inputs: info ? Object.keys(info.inputs) : [],
                outputs: info ? Object.keys(info.outputs) : [],
                inputTypes: info?.inputs ?? {},
                outputTypes: info?.outputs ?? {},
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
            data: {},
          }))
        );
      } catch (err) {
        console.error("[pipeline] Init failed:", err);
      }
    })();
  }, [components, componentMap, setNodes, setEdges]);

  // Update node and edge data with metrics
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
    setEdges((prev) =>
      prev.map((e) => {
        const slot = parseSlot(e.sourceHandle);
        const ch = metrics.nodes[e.source]?.channels?.[slot];
        const sub = ch?.subscribers?.[e.target];
        return {
          ...e,
          data: { byteDelta: sub?.byte_count_delta ?? 0 },
        };
      })
    );
  }, [metrics, setNodes, setEdges]);

  // Wrap node changes — detect removals and call backend
  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      const removals = changes.filter((c) => c.type === "remove");
      onNodesChangeRaw(changes);

      for (const r of removals) {
        if (r.type === "remove") {
          // Configuring nodes are local-only — no backend call needed
          if (r.id.startsWith("configuring-")) continue;

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
          { ...connection, type: "pipeline", data: {} },
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

  const createPipelineNode = useCallback(
    (item: ComponentInfo, position: { x: number; y: number }, config?: Record<string, unknown>) => {
      apiCreateNode(item.name, config)
        .then((res) => {
          const newNode: Node<PipelineNodeData> = {
            id: res.id,
            type: "pipeline",
            position,
            data: {
              label: item.name,
              category: item.category,
              inputs: Object.keys(item.inputs),
              outputs: Object.keys(item.outputs),
              inputTypes: item.inputs,
              outputTypes: item.outputs,
              status: "startup",
              nodeMetrics: null,
            },
          };
          setNodes((nds) => [...nds, newNode]);
        })
        .catch(console.error);
    },
    [setNodes],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData("application/pipeline-node");
      if (!raw) return;

      const item = JSON.parse(raw) as ComponentInfo;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });

      // Check if any init param has configurable properties
      const hasConfig = Object.values(item.init).some((schema) => {
        if (!schema || typeof schema !== "object") return false;
        const s = schema as Record<string, unknown>;
        if (s.properties) return true;
        if (s.$ref) return true;
        if (Array.isArray(s.anyOf)) {
          return (s.anyOf as Record<string, unknown>[]).some(
            (branch) => branch.type === "object" || branch.$ref,
          );
        }
        return false;
      });

      if (!hasConfig) {
        createPipelineNode(item, position);
        return;
      }

      // Add a temporary "configuring" node with the form
      const tempId = `configuring-${Date.now()}`;
      const configuringNode: Node = {
        id: tempId,
        type: "configuring",
        position,
        data: {
          componentInfo: item,
          onConfirm: (config: Record<string, unknown>) => {
            setNodes((nds) => nds.filter((n) => n.id !== tempId));
            createPipelineNode(item, position, config);
          },
          onCancel: () => {
            setNodes((nds) => nds.filter((n) => n.id !== tempId));
          },
        },
      };
      setNodes((nds) => [...nds, configuringNode]);
    },
    [screenToFlowPosition, setNodes, createPipelineNode],
  );

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      <PipelineCanvas
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
      />
      <NodeSidebar components={components} />
      <MetricsOverlay
        connected={connected}
        metrics={metrics}
        onOpenDashboard={() => setMetricsOpen(true)}
      />
      {metricsOpen && (
        <MetricsDashboard
          connected={connected}
          history={history}
          componentMap={componentMap}
          onClose={() => setMetricsOpen(false)}
        />
      )}
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

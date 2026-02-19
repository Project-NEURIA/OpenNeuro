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
  type OnNodeDrag,
} from "@xyflow/react";
import { Home } from "lucide-react";
import { PipelineCanvas } from "@/components/pipeline/PipelineCanvas";
import { NodeSidebar } from "@/components/pipeline/NodeSidebar";
import { MetricsOverlay } from "@/components/pipeline/MetricsOverlay";
import { MetricsDashboard } from "@/components/metrics/MetricsDashboard";
import { ProjectChooser } from "@/components/project/ProjectChooser";
import { usePipelineData, type PipelineNodeData } from "@/hooks/usePipelineData";
import { useComponents } from "@/hooks/useComponents";
import { useMetricsHistory } from "@/hooks/useMetricsHistory";
import { layoutNodes } from "@/lib/layout";
import {
  fetchNodes as apiFetchNodes,
  fetchEdges as apiFetchEdges,
  createNode as apiCreateNode,
  updateNode as apiUpdateNode,
  deleteNode as apiDeleteNode,
  createEdge as apiCreateEdge,
  deleteEdge as apiDeleteEdge,
  fetchCurrentProject,
  startProject as apiStartProject,
  closeProject as apiCloseProject,
  saveGraph,
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

function AppInner({
  projectName,
  onGoHome,
}: {
  projectName: string;
  onGoHome: () => void;
}) {
  const components = useComponents();
  const { connected, metrics, componentMap } = usePipelineData(components);

  const [metricsOpen, setMetricsOpen] = useState(false);
  const history = useMetricsHistory(metrics);

  const [nodes, setNodes, onNodesChangeRaw] = useNodesState<Node>([] as Node[]);
  const [edges, setEdges, onEdgesChangeRaw] = useEdgesState<Edge>([] as Edge[]);
  const initialized = useRef(false);
  const { screenToFlowPosition } = useReactFlow();

  const triggerSave = useCallback(() => {
    saveGraph().catch(console.error);
  }, []);

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

        const allZero = backendNodes.every((n) => n.x === 0 && n.y === 0);

        let posMap: Map<string, { x: number; y: number }>;
        if (allZero && backendNodes.length > 0) {
          const nodeSpecs = backendNodes.map((n) => ({ id: n.id, type: n.type }));
          const edgeSpecs = backendEdges.map((e) => ({
            source: e.source_node,
            target: e.target_node,
          }));
          const positions = layoutNodes(nodeSpecs, edgeSpecs);
          posMap = new Map(positions.map((p) => [p.id, p]));
        } else {
          posMap = new Map(backendNodes.map((n) => [n.id, { x: n.x, y: n.y }]));
        }

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
          }),
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
          })),
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
      }),
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
      }),
    );
  }, [metrics, setNodes, setEdges]);

  // Wrap node changes — detect removals and call backend
  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      const removals = changes.filter((c) => c.type === "remove");
      onNodesChangeRaw(changes);

      for (const r of removals) {
        if (r.type === "remove") {
          if (r.id.startsWith("configuring-")) continue;

          setEdges((currentEdges) => {
            for (const e of currentEdges) {
              if (e.source === r.id || e.target === r.id) {
                deleteEdgeFromReactFlow(e);
              }
            }
            return currentEdges.filter(
              (e) => e.source !== r.id && e.target !== r.id,
            );
          });
          apiDeleteNode(r.id)
            .then(() => triggerSave())
            .catch(console.error);
        }
      }
    },
    [onNodesChangeRaw, setEdges, triggerSave],
  );

  // Wrap edge changes — detect removals and call backend
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      const hasRemovals = changes.some((c) => c.type === "remove");
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
      if (hasRemovals) triggerSave();
    },
    [onEdgesChangeRaw, setEdges, triggerSave],
  );

  // Handle new edge connections
  const onConnect: OnConnect = useCallback(
    (connection) => {
      setEdges((eds) =>
        addEdge({ ...connection, type: "pipeline", data: {} }, eds),
      );
      if (connection.source && connection.target) {
        const sourceSlot = parseSlot(connection.sourceHandle);
        const targetSlot = parseSlot(connection.targetHandle);
        apiCreateEdge(
          connection.source,
          sourceSlot,
          connection.target,
          targetSlot,
        )
          .then(() => triggerSave())
          .catch(console.error);
      }
    },
    [setEdges, triggerSave],
  );

  const onNodeDragStop: OnNodeDrag = useCallback(
    (_event, node) => {
      apiUpdateNode(node.id, { x: node.position.x, y: node.position.y }).catch(
        console.error,
      );
      triggerSave();
    },
    [triggerSave],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const createPipelineNode = useCallback(
    (
      item: ComponentInfo,
      position: { x: number; y: number },
      initArgs?: Record<string, unknown>,
    ) => {
      apiCreateNode(item.name, initArgs)
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
          triggerSave();
        })
        .catch(console.error);
    },
    [setNodes, triggerSave],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const raw = e.dataTransfer.getData("application/pipeline-node");
      if (!raw) return;

      const item = JSON.parse(raw) as ComponentInfo;
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });

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

      const tempId = `configuring-${Date.now()}`;
      const configuringNode: Node = {
        id: tempId,
        type: "configuring",
        position,
        data: {
          componentInfo: item,
          onConfirm: (initArgs: Record<string, unknown>) => {
            setNodes((nds) => nds.filter((n) => n.id !== tempId));
            createPipelineNode(item, position, initArgs);
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

  const handleGoHome = useCallback(async () => {
    await apiCloseProject();
    onGoHome();
  }, [onGoHome]);

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
        onNodeDragStop={onNodeDragStop}
      />
      <NodeSidebar components={components} />

      {/* Home button */}
      <button
        onClick={handleGoHome}
        className="absolute top-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-2 rounded-lg bg-[var(--glass)] px-3 py-1.5 text-sm text-[var(--muted-foreground)] backdrop-blur transition-colors hover:bg-[var(--glass-hover)] hover:text-[var(--foreground)]"
        title="Back to projects"
      >
        <Home size={16} />
        <span>{projectName}</span>
      </button>

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
  // undefined = loading, null = no project, string = project name
  const [currentProject, setCurrentProject] = useState<
    string | null | undefined
  >(undefined);
  const [showChooser, setShowChooser] = useState(false);

  useEffect(() => {
    fetchCurrentProject()
      .then(async ({ current_project }) => {
        if (current_project) {
          await apiStartProject(current_project);
          setCurrentProject(current_project);
        } else {
          setCurrentProject(null);
          setShowChooser(true);
        }
      })
      .catch(() => {
        setCurrentProject(null);
        setShowChooser(true);
      });
  }, []);

  // Loading
  if (currentProject === undefined && !showChooser) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-[var(--background)]">
        <span className="text-[var(--muted-foreground)]">Loading...</span>
      </div>
    );
  }

  // Project chooser
  if (showChooser || currentProject === null) {
    return (
      <ProjectChooser
        hadProject={currentProject !== null}
        onOpen={(name) => {
          setCurrentProject(name);
          setShowChooser(false);
        }}
        onCancel={() => setShowChooser(false)}
      />
    );
  }

  // Editor
  return (
    <ReactFlowProvider>
      <AppInner
        key={currentProject}
        projectName={currentProject!}
        onGoHome={() => {
          setCurrentProject(null);
          setShowChooser(true);
        }}
      />
    </ReactFlowProvider>
  );
}

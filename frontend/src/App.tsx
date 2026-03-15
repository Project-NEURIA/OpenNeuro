import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { SplashScreen } from "@/components/SplashScreen";
import { GraphCanvas } from "@/components/graph/GraphCanvas";
import { NodeSidebar } from "@/components/graph/NodeSidebar";
import { MetricsOverlay } from "@/components/graph/MetricsOverlay";
import { MetricsDashboard } from "@/components/metrics/MetricsDashboard";
import { ProjectChooser } from "@/components/project/ProjectChooser";
import { UIChannelProvider } from "@/contexts/UIChannelContext";
import { useGraphData, type GraphNodeData } from "@/hooks/useGraphData";
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
  createSubgraph as apiCreateSubgraph,
  ungroupNode as apiUngroupNode,
  fetchCurrentProject,
  fetchIsType,
  fetchProjects,
  startProject as apiStartProject,
  closeProject as apiCloseProject,
  saveGraph,
  type NodeResponse,
  type ProjectSummary,
} from "@/lib/api";
import { parseSlotType, type ComponentInfo, type Graph, type GraphEdge, type SlotType } from "@/lib/types";
import { checkTypes, collectLeafNames, typeToString, warmSubtypeCache } from "@/lib/typecheck";

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

/** Build a ReactFlow node from a backend NodeResponse. */
function toReactFlowNode(
  n: NodeResponse,
  position: { x: number; y: number },
  componentMap: Record<string, ComponentInfo>,
  componentTypeInfo: { inputs: Record<string, Record<string, SlotType>>; outputs: Record<string, Record<string, SlotType>> },
): Node<GraphNodeData> {
  const isComposite = n.is_composite;
  const info = componentMap[n.type];

  // For composite nodes, derive ports from the response; for regular, from the component registry
  const inputs = isComposite
    ? Object.keys(n.inputs ?? {})
    : info ? Object.keys(info.inputs) : [];
  const outputs = isComposite
    ? Object.keys(n.outputs ?? {})
    : info ? Object.keys(info.outputs) : [];

  const compInputs = n.inputs ?? {};
  const compOutputs = n.outputs ?? {};
  const inputTypes = isComposite
    ? Object.fromEntries(Object.entries(compInputs).map(([k, v]) => [k, parseSlotType(v)]))
    : componentTypeInfo.inputs[n.type] ?? {};
  const outputTypes = isComposite
    ? Object.fromEntries(Object.entries(compOutputs).map(([k, v]) => [k, parseSlotType(v)]))
    : componentTypeInfo.outputs[n.type] ?? {};

  return {
    id: n.id,
    type: "graph",
    position,
    data: {
      label: n.label ?? (isComposite ? "Subgraph" : n.type),
      category: isComposite ? "composite" : (info?.category ?? "conduit"),
      inputs,
      outputs,
      inputTypes,
      outputTypes,
      status: n.status,
      nodeMetrics: null,
      ui_inputs: info?.ui_inputs ?? {},
      ui_outputs: info?.ui_outputs ?? {},
    } satisfies GraphNodeData,
  };
}

function AppInner({
  projectName,
  onGoHome,
}: {
  projectName: string;
  onGoHome: () => void;
}) {
  const components = useComponents();
  const { connected, metrics, componentMap } = useGraphData(components);

  const [metricsOpen, setMetricsOpen] = useState(false);
  const history = useMetricsHistory(metrics);

  // Context menu state
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeIds: string[];
    isComposite: boolean;
    naming: boolean;
  } | null>(null);
  const [subgraphName, setSubgraphName] = useState("Subgraph");
  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);

  // Fetch projects for sidebar
  useEffect(() => {
    fetchProjects().then(setProjects).catch(console.error);
  }, []);

  const [nodes, setNodes, onNodesChangeRaw] = useNodesState<Node>([] as Node[]);
  const [edges, setEdges, onEdgesChangeRaw] = useEdgesState<Edge>([] as Edge[]);
  const initialized = useRef(false);
  const { screenToFlowPosition } = useReactFlow();

  const triggerSave = useCallback(() => {
    saveGraph().catch(console.error);
  }, []);

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  const componentTypeInfo = useMemo(() => {
    const inputs: Record<string, Record<string, SlotType>> = {};
    const outputs: Record<string, Record<string, SlotType>> = {};
    for (const c of components) {
      inputs[c.name] = Object.fromEntries(
        Object.entries(c.inputs).map(([k, v]) => [k, parseSlotType(v)]),
      );
      outputs[c.name] = Object.fromEntries(
        Object.entries(c.outputs).map(([k, v]) => [k, parseSlotType(v)]),
      );
    }
    return { inputs, outputs };
  }, [components]);

  const [concreteTypes, setConcreteTypes] = useState<Set<string>>(new Set());

  useEffect(() => {
    const leafNames = new Set<string>();
    for (const slots of [...Object.values(componentTypeInfo.inputs), ...Object.values(componentTypeInfo.outputs)]) {
      for (const slot of Object.values(slots)) {
        for (const name of collectLeafNames(slot.name)) leafNames.add(name);
      }
    }
    if (leafNames.size === 0) return;
    Promise.all(
      [...leafNames].map(async (name) => [name, await fetchIsType(name)] as const),
    ).then(async (results) => {
      const concrete = new Set(results.filter(([, ok]) => ok).map(([name]) => name));
      await warmSubtypeCache(concrete);
      setConcreteTypes(concrete);
    });
  }, [componentTypeInfo]);

  const runTypeCheck = useCallback(() => {
    const currentNodes = nodesRef.current;

    setEdges((currentEdges) => {
      const graph: Graph = {
        nodes: Object.fromEntries(
          currentNodes
            .filter((n) => n.type === "graph")
            .map((n) => {
              const data = n.data as GraphNodeData;
              return [n.id, { type: data.label, init_args: {}, x: n.position.x, y: n.position.y }];
            }),
        ),
        edges: currentEdges.map((e): GraphEdge => ({
          source_node: e.source,
          source_slot: parseSlot(e.sourceHandle),
          target_node: e.target,
          target_slot: parseSlot(e.targetHandle),
        })),
      };

      const { errors, types } = checkTypes(graph, componentTypeInfo.inputs, componentTypeInfo.outputs, concreteTypes);

      // Build per-node resolved type maps (only for vars that resolved to non-var types)
      const nodeResolved = new Map<string, Record<string, string>>();
      for (const [key, typ] of types) {
        if (typ.kind === "var") continue;
        // key format: "nodeId.in.slot" or "nodeId.out.slot"
        const firstDot = key.indexOf(".");
        const nodeId = key.slice(0, firstDot);
        const rest = key.slice(firstDot + 1); // "in.slot" or "out.slot"
        let resolved = nodeResolved.get(nodeId);
        if (!resolved) {
          resolved = {};
          nodeResolved.set(nodeId, resolved);
        }
        resolved[rest] = typeToString(typ);
      }

      // Update nodes with resolved types
      setNodes((prev) =>
        prev.map((n) => {
          const r = nodeResolved.get(n.id);
          if (!r) return n;
          return { ...n, data: { ...(n.data as GraphNodeData), resolvedTypes: r } };
        }),
      );

      const errorMap = new Map<string, string>();
      for (const err of errors) {
        if (err.constraint.origin.kind === "edge") {
          const { sourceNode, sourceSlot, targetNode, targetSlot } = err.constraint.origin;
          const edgeId = `${sourceNode}:${sourceSlot}->${targetNode}:${targetSlot}`;
          errorMap.set(edgeId, `${typeToString(err.left)} ≠ ${typeToString(err.right)}`);
        }
      }

      return currentEdges.map((e) => {
        const srcSlot = parseSlot(e.sourceHandle);
        const tgtSlot = parseSlot(e.targetHandle);
        const key = `${e.source}:${srcSlot}->${e.target}:${tgtSlot}`;
        return {
          ...e,
          data: {
            ...(e.data as Record<string, unknown>),
            typeError: errorMap.get(key) || undefined,
          },
        };
      });
    });
  }, [setEdges, setNodes, componentTypeInfo, concreteTypes]);

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
            return toReactFlowNode(n, pos, componentMap, componentTypeInfo);
          }),
        );

        setEdges(
          backendEdges.map((e) => ({
            id: `${e.source_node}:${e.source_slot}->${e.target_node}:${e.target_slot}`,
            source: e.source_node,
            sourceHandle: `out-${e.source_slot}`,
            target: e.target_node,
            targetHandle: `in-${e.target_slot}`,
            type: "graph",
            data: {},
          })),
        );

        runTypeCheck();
      } catch (err) {
        console.error("[graph] Init failed:", err);
      }
    })();
  }, [components, componentMap, setNodes, setEdges, runTypeCheck]);

  // Update node and edge data with metrics
  useEffect(() => {
    if (!initialized.current || !metrics) return;
    setNodes((prev) =>
      prev.map((n) => {
        const nodeMetrics = metrics.nodes[n.id] ?? null;
        const status = nodeMetrics?.status ?? (n.data as GraphNodeData).status;

        return {
          ...n,
          data: {
            ...(n.data as GraphNodeData),
            status,
            nodeMetrics,
          },
        };
      }),
    );
    setEdges((prev) =>
      prev.map((e) => {
        const targetSlot = parseSlot(e.targetHandle);
        const recv = metrics.nodes[e.target]?.receivers?.[targetSlot];
        return {
          ...e,
          data: { ...(e.data as Record<string, unknown>), byteDelta: recv?.byte_count_delta ?? 0 },
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
            .then(() => {
              runTypeCheck();
              triggerSave();
            })
            .catch(console.error);
        }
      }
    },
    [onNodesChangeRaw, setEdges, triggerSave, runTypeCheck],
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
      if (hasRemovals) {
        runTypeCheck();
        triggerSave();
      }
    },
    [onEdgesChangeRaw, setEdges, triggerSave, runTypeCheck],
  );

  // Handle new edge connections
  const onConnect: OnConnect = useCallback(
    (connection) => {
      setEdges((eds) =>
        addEdge({ ...connection, type: "graph", data: {} }, eds),
      );
      runTypeCheck();
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
    [setEdges, triggerSave, runTypeCheck],
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

  const createGraphNode = useCallback(
    (
      item: ComponentInfo,
      position: { x: number; y: number },
      initArgs?: Record<string, unknown>,
    ) => {
      apiCreateNode(item.name, initArgs)
        .then((res) => {
          const newNode: Node<GraphNodeData> = {
            id: res.id,
            type: "graph",
            position,
            data: {
              label: item.name,
              category: item.category,
              inputs: Object.keys(item.inputs),
              outputs: Object.keys(item.outputs),
              inputTypes: componentTypeInfo.inputs[item.name] ?? {},
              outputTypes: componentTypeInfo.outputs[item.name] ?? {},
              status: "startup",
              nodeMetrics: null,
              ui_inputs: item.ui_inputs ?? {},
              ui_outputs: item.ui_outputs ?? {},
            },
          };
          setNodes((nds) => [...nds, newNode]);
          runTypeCheck();
          triggerSave();
        })
        .catch(console.error);
    },
    [setNodes, triggerSave, runTypeCheck],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();

      // Handle project drops — create a composite node using project name as type
      const projectName = e.dataTransfer.getData("application/project-node");
      if (projectName) {
        const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });
        apiCreateNode(projectName)
          .then((res) => {
            const newNode = toReactFlowNode(res, position, componentMap, componentTypeInfo);
            setNodes((nds) => [...nds, newNode]);
            runTypeCheck();
            triggerSave();
          })
          .catch(console.error);
        return;
      }

      const raw = e.dataTransfer.getData("application/graph-node");
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
        createGraphNode(item, position);
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
            createGraphNode(item, position, initArgs);
          },
          onCancel: () => {
            setNodes((nds) => nds.filter((n) => n.id !== tempId));
          },
        },
      };
      setNodes((nds) => [...nds, configuringNode]);
    },
    [screenToFlowPosition, setNodes, createGraphNode],
  );

  /** Re-fetch full graph state from backend and rebuild ReactFlow nodes/edges. */
  const refreshGraph = useCallback(async () => {
    const [backendNodes, backendEdges] = await Promise.all([
      apiFetchNodes(),
      apiFetchEdges(),
    ]);
    setNodes(
      backendNodes.map((n) =>
        toReactFlowNode(n, { x: n.x, y: n.y }, componentMap, componentTypeInfo),
      ),
    );
    setEdges(
      backendEdges.map((e) => ({
        id: `${e.source_node}:${e.source_slot}->${e.target_node}:${e.target_slot}`,
        source: e.source_node,
        sourceHandle: `out-${e.source_slot}`,
        target: e.target_node,
        targetHandle: `in-${e.target_slot}`,
        type: "graph",
        data: {},
      })),
    );
    runTypeCheck();
  }, [componentMap, componentTypeInfo, setNodes, setEdges, runTypeCheck]);

  const onSelectionChange = useCallback(
    ({ nodes: sel }: { nodes: Node[] }) => {
      setSelectedNodeIds(sel.filter((n) => n.type === "graph").map((n) => n.id));
    },
    [],
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: Node) => {
      event.preventDefault();
      const data = node.data as GraphNodeData;
      const isComposite = data.category === "composite";
      // If node is part of selection, use full selection; otherwise just this node
      const ids = selectedNodeIds.includes(node.id) ? selectedNodeIds : [node.id];
      setContextMenu({
        x: event.clientX,
        y: event.clientY,
        nodeIds: ids,
        isComposite,
        naming: false,
      });
    },
    [selectedNodeIds],
  );

  const handleGroupClick = useCallback(() => {
    if (!contextMenu || contextMenu.nodeIds.length < 2) return;
    setSubgraphName("Subgraph");
    setContextMenu({ ...contextMenu, naming: true });
  }, [contextMenu]);

  const handleGroupConfirm = useCallback(async () => {
    if (!contextMenu) return;
    const name = subgraphName.trim() || "Subgraph";
    const ids = contextMenu.nodeIds;
    setContextMenu(null);
    try {
      await apiCreateSubgraph(ids, name);
      await refreshGraph();
      triggerSave();
    } catch (err) {
      console.error("[graph] Group failed:", err);
    }
  }, [contextMenu, subgraphName, refreshGraph, triggerSave]);

  const handleUngroup = useCallback(async () => {
    if (!contextMenu || contextMenu.nodeIds.length !== 1) return;
    setContextMenu(null);
    try {
      await apiUngroupNode(contextMenu.nodeIds[0]!);
      await refreshGraph();
      triggerSave();
    } catch (err) {
      console.error("[graph] Ungroup failed:", err);
    }
  }, [contextMenu, refreshGraph, triggerSave]);

  const handleGoHome = useCallback(async () => {
    await apiCloseProject();
    onGoHome();
  }, [onGoHome]);

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      <GraphCanvas
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeDragStop={onNodeDragStop}
        onSelectionChange={onSelectionChange}
        onNodeContextMenu={onNodeContextMenu}
        onPaneClick={() => setContextMenu(null)}
      />
      <NodeSidebar components={components} projects={projects} currentProject={projectName} />

      {/* Context menu */}
      {contextMenu && (
        <div
          className="fixed z-50 rounded-lg border border-white/10 bg-[var(--glass)] backdrop-blur-xl shadow-lg py-1 min-w-[200px]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          {contextMenu.isComposite && contextMenu.nodeIds.length === 1 && (
            <button
              onClick={handleUngroup}
              className="w-full px-4 py-2 text-left text-sm text-[var(--foreground)] hover:bg-white/[0.06] transition-colors"
            >
              Ungroup subgraph
            </button>
          )}
          {!contextMenu.isComposite && contextMenu.nodeIds.length >= 2 && !contextMenu.naming && (
            <button
              onClick={handleGroupClick}
              className="w-full px-4 py-2 text-left text-sm text-[var(--foreground)] hover:bg-white/[0.06] transition-colors"
            >
              Group into subgraph
            </button>
          )}
          {!contextMenu.isComposite && contextMenu.nodeIds.length >= 2 && contextMenu.naming && (
            <div className="px-3 py-2 flex flex-col gap-2">
              <label className="text-[11px] font-medium text-[var(--muted-foreground)] uppercase tracking-wider">
                Name
              </label>
              <input
                autoFocus
                type="text"
                value={subgraphName}
                onChange={(e) => setSubgraphName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleGroupConfirm();
                  if (e.key === "Escape") setContextMenu(null);
                }}
                className="rounded-md bg-black/40 border border-white/10 px-2.5 py-1.5 text-sm text-[var(--foreground)] font-mono focus:outline-none focus:border-white/25"
              />
              <button
                onClick={handleGroupConfirm}
                className="rounded-md bg-white/[0.08] border border-white/10 px-3 py-1.5 text-sm text-[var(--foreground)] hover:bg-white/[0.12] transition-colors"
              >
                Create
              </button>
            </div>
          )}
          {contextMenu.nodeIds.length < 2 && !contextMenu.isComposite && (
            <div className="px-4 py-2 text-sm text-[var(--muted-foreground)]">
              Select 2+ nodes to group
            </div>
          )}
        </div>
      )}

      {/* Window drag region */}
      <div data-tauri-drag-region className="absolute top-0 left-0 right-0 h-[15px] z-50" />

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
  const [splashStatus, setSplashStatus] = useState("Connecting to backend...");

  useEffect(() => {
    let cancelled = false;

    async function pollBackend() {
      // Poll until backend is ready
      while (!cancelled) {
        try {
          const { current_project } = await fetchCurrentProject();
          if (cancelled) return;
          if (current_project) {
            setSplashStatus("Loading project...");
            await apiStartProject(current_project);
            setCurrentProject(current_project);
          } else {
            setCurrentProject(null);
            setShowChooser(true);
          }
          return;
        } catch {
          // Backend not ready yet, retry
          await new Promise((r) => setTimeout(r, 500));
        }
      }
    }

    pollBackend();
    return () => { cancelled = true; };
  }, []);

  // Splash screen while waiting for backend
  if (currentProject === undefined && !showChooser) {
    return <SplashScreen status={splashStatus} />;
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
      <UIChannelProvider>
        <AppInner
          key={currentProject}
          projectName={currentProject!}
          onGoHome={() => {
            setCurrentProject(null);
            setShowChooser(true);
          }}
        />
      </UIChannelProvider>
    </ReactFlowProvider>
  );
}

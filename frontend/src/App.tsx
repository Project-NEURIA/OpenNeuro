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
} from "@xyflow/react";
import { PipelineCanvas } from "@/components/pipeline/PipelineCanvas";
import { NodeSidebar } from "@/components/pipeline/NodeSidebar";
import { MetricsOverlay } from "@/components/pipeline/MetricsOverlay";
import { usePipelineData, type PipelineNodeData } from "@/hooks/usePipelineData";

function AppInner() {
  const {
    defaultNodes,
    defaultEdges,
    connected,
    metrics,
    syncPipeline,
    typeLabels,
  } = usePipelineData();

  const [nodes, setNodes, onNodesChangeRaw] = useNodesState([]);
  const [edges, setEdges, onEdgesChangeRaw] = useEdgesState([]);
  const initialized = useRef(false);
  const { screenToFlowPosition } = useReactFlow();

  // Initialize graph state from defaults (once)
  useEffect(() => {
    if (initialized.current || defaultNodes.length === 0) return;
    initialized.current = true;
    setNodes(defaultNodes);
    setEdges(defaultEdges);
  }, [defaultNodes, defaultEdges, setNodes, setEdges]);

  // Update metrics data on existing nodes without replacing positions
  useEffect(() => {
    if (!initialized.current) return;
    setNodes((prev) =>
      prev.map((n) => {
        const fresh = defaultNodes.find((d) => d.id === n.id);
        if (!fresh) return n;
        return { ...n, data: fresh.data };
      })
    );
  }, [defaultNodes, setNodes]);

  // Update edge data (metrics) without replacing structure
  useEffect(() => {
    if (!initialized.current) return;
    setEdges((prev) =>
      prev.map((e) => {
        const fresh = defaultEdges.find((d) => d.id === e.id);
        if (!fresh) return e;
        return { ...e, data: fresh.data };
      })
    );
  }, [defaultEdges, setEdges]);

  // Wrap node changes — detect removals and sync
  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      const hasRemoval = changes.some((c) => c.type === "remove");
      onNodesChangeRaw(changes);
      if (hasRemoval) {
        // Sync after state settles
        setNodes((current) => {
          // Also remove edges that reference removed nodes
          const nodeIds = new Set(current.map((n) => n.id));
          setEdges((currentEdges) => {
            const filtered = currentEdges.filter(
              (e) => nodeIds.has(e.source) && nodeIds.has(e.target)
            );
            // Schedule sync with the cleaned-up state
            setTimeout(() => syncPipeline(current as Node<PipelineNodeData>[], filtered), 0);
            return filtered;
          });
          return current;
        });
      }
    },
    [onNodesChangeRaw, setNodes, setEdges, syncPipeline]
  );

  // Wrap edge changes — detect removals and sync
  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      const hasRemoval = changes.some((c) => c.type === "remove");
      onEdgesChangeRaw(changes);
      if (hasRemoval) {
        setEdges((current) => {
          setNodes((currentNodes) => {
            setTimeout(() => syncPipeline(currentNodes as Node<PipelineNodeData>[], current), 0);
            return currentNodes;
          });
          return current;
        });
      }
    },
    [onEdgesChangeRaw, setEdges, setNodes, syncPipeline]
  );

  // Handle new edge connections
  const onConnect: OnConnect = useCallback(
    (connection) => {
      setEdges((eds) => {
        const updated = addEdge(
          { ...connection, type: "pipeline", data: { topicName: "", msgPerSec: 0 } },
          eds
        );
        setNodes((currentNodes) => {
          setTimeout(() => syncPipeline(currentNodes as Node<PipelineNodeData>[], updated), 0);
          return currentNodes;
        });
        return updated;
      });
    },
    [setEdges, setNodes, syncPipeline]
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

      const item = JSON.parse(raw) as { name: string };
      const info = typeLabels[item.name];
      if (!info) return;

      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY });

      const newNode: Node<PipelineNodeData> = {
        id: item.name,
        type: "pipeline",
        position,
        data: {
          label: item.name,
          category: info.category,
          input_type: info.input,
          output_type: info.output,
          status: "idle",
          topicMetrics: null,
          nodeMetrics: null,
        },
      };

      setNodes((nds) => {
        // Don't add if already exists
        if (nds.some((n) => n.id === newNode.id)) return nds;
        const updated = [...nds, newNode];
        setEdges((currentEdges) => {
          setTimeout(() => syncPipeline(updated as Node<PipelineNodeData>[], currentEdges), 0);
          return currentEdges;
        });
        return updated;
      });
    },
    [screenToFlowPosition, typeLabels, setNodes, setEdges, syncPipeline]
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <NodeSidebar />
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

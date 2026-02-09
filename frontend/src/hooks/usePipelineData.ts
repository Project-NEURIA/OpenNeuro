import { useEffect, useRef, useCallback, useMemo } from "react";
import type { Node, Edge } from "@xyflow/react";
import { postPipeline } from "@/lib/api";
import { layoutNodes } from "@/lib/layout";
import { useWebSocket } from "./useWebSocket";
import type {
  PipelineNode,
  PipelineEdge,
  MetricsSnapshot,
  TopicMetrics,
  NodeMetrics,
} from "@/lib/types";

export interface PipelineNodeData extends Record<string, unknown> {
  label: string;
  category: PipelineNode["category"];
  input_type: string | null;
  output_type: string | null;
  status: PipelineNode["status"];
  topicMetrics: TopicMetrics | null;
  nodeMetrics: NodeMetrics | null;
}

// Static type map (mirrors backend _TYPE_LABELS)
const TYPE_LABELS: Record<string, { category: PipelineNode["category"]; input: string | null; output: string | null }> = {
  Microphone: { category: "source", input: null, output: "bytes" },
  VAD:        { category: "conduit", input: "bytes", output: "bytes" },
  ASR:        { category: "conduit", input: "bytes", output: "str" },
  LLM:        { category: "conduit", input: "str", output: "str" },
  TTS:        { category: "conduit", input: "str", output: "bytes" },
  STS:        { category: "conduit", input: "bytes", output: "bytes" },
  Speaker:    { category: "sink", input: "bytes", output: null },
};

// Default pipeline: Mic → STS → Speaker
const DEFAULT_PIPELINE_NODES: PipelineNode[] = [
  { id: "Microphone", name: "Microphone", category: "source", input_type: null, output_type: "bytes", status: "idle" },
  { id: "STS",        name: "STS",        category: "conduit", input_type: "bytes", output_type: "bytes", status: "idle" },
  { id: "Speaker",    name: "Speaker",    category: "sink", input_type: "bytes", output_type: null, status: "idle" },
];

const DEFAULT_PIPELINE_EDGES: PipelineEdge[] = [
  { id: "Mic->STS", source: "Microphone", target: "STS", topic_name: "Microphone_out" },
  { id: "STS->Speaker",    source: "STS",        target: "Speaker", topic_name: "STS_out" },
];

export function usePipelineData() {
  const { data: metrics, connected } =
    useWebSocket<MetricsSnapshot>("/api/ws/metrics");

  const hasSynced = useRef(false);

  // POST default pipeline on first mount
  useEffect(() => {
    if (hasSynced.current) return;
    hasSynced.current = true;
    postPipeline({
      nodes: DEFAULT_PIPELINE_NODES.map((n) => n.name),
      edges: DEFAULT_PIPELINE_EDGES.map((e) => ({ source: e.source, target: e.target })),
    }).catch((err) => console.error("[pipeline] Initial sync failed:", err));
  }, []);

  const defaultNodes = useMemo(() => {
    const positions = layoutNodes(DEFAULT_PIPELINE_NODES, DEFAULT_PIPELINE_EDGES);
    const posMap = new Map(positions.map((p) => [p.id, p]));

    return DEFAULT_PIPELINE_NODES.map((n): Node<PipelineNodeData> => {
      const pos = posMap.get(n.id) ?? { x: 0, y: 0 };
      const topicName = `${n.name}_out`;
      const topicMetrics = metrics?.topics[topicName] ?? null;
      const nodeMetrics = metrics?.nodes[n.name] ?? null;
      const status = nodeMetrics?.status ?? n.status;

      return {
        id: n.id,
        type: "pipeline",
        position: { x: pos.x, y: pos.y },
        data: {
          label: n.name,
          category: n.category,
          input_type: n.input_type,
          output_type: n.output_type,
          status,
          topicMetrics,
          nodeMetrics,
        },
      };
    });
  }, [metrics]);

  const defaultEdges = useMemo((): Edge[] => {
    return DEFAULT_PIPELINE_EDGES.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: "pipeline",
      data: {
        topicName: e.topic_name,
        msgPerSec: metrics?.topics[e.topic_name]?.msg_per_sec ?? 0,
      },
    }));
  }, [metrics]);

  // Sync function: POST current graph to backend
  const syncPipeline = useCallback(
    (nodes: Node<PipelineNodeData>[], edges: Edge[]) => {
      const nodeNames = nodes.map((n) => n.data.label);
      const edgeList = edges.map((e) => ({ source: e.source, target: e.target }));
      postPipeline({ nodes: nodeNames, edges: edgeList }).catch((err) =>
        console.error("[pipeline] Sync failed:", err)
      );
    },
    []
  );

  return {
    defaultNodes,
    defaultEdges,
    connected,
    metrics,
    syncPipeline,
    typeLabels: TYPE_LABELS,
  };
}

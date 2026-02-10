import { useMemo } from "react";
import { useSSE } from "./useSSE";
import type { ComponentInfo, MetricsSnapshot, TopicMetrics, NodeMetrics } from "@/lib/types";

export interface PipelineNodeData extends Record<string, unknown> {
  label: string;
  category: "source" | "conduit" | "sink";
  input_type: string | null;
  output_type: string | null;
  status: string;
  topicMetrics: TopicMetrics | null;
  nodeMetrics: NodeMetrics | null;
}

export function usePipelineData(components: ComponentInfo[]) {
  const { data: metrics, connected } =
    useSSE<MetricsSnapshot>("/metrics");

  const componentMap = useMemo(() => {
    const map: Record<string, ComponentInfo> = {};
    for (const c of components) map[c.name] = c;
    return map;
  }, [components]);

  return {
    connected,
    metrics,
    componentMap,
  };
}

import { useMemo } from "react";
import { useSSE } from "./useSSE";
import type { ComponentInfo, MetricsSnapshot, NodeMetrics } from "@/lib/types";

export interface PipelineNodeData extends Record<string, unknown> {
  label: string;
  category: "source" | "conduit" | "sink";
  inputs: string[];
  outputs: string[];
  inputTypes: Record<string, string>;
  outputTypes: Record<string, string>;
  status: string;
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

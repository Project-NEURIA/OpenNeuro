import { useMemo } from "react";
import { useSSE } from "./useSSE";
import type { ComponentInfo, MetricsSnapshot, NodeMetrics, SlotType } from "@/lib/types";

export interface GraphNodeData extends Record<string, unknown> {
  label: string;
  category: "source" | "conduit" | "sink" | "composite";
  inputs: string[];
  outputs: string[];
  inputTypes: Record<string, SlotType>;
  outputTypes: Record<string, SlotType>;
  status: string;
  nodeMetrics: NodeMetrics | null;
  resolvedTypes?: Record<string, string>;
  ui_inputs: Record<string, string>;
  ui_outputs: Record<string, string>;
}

export function useGraphData(components: ComponentInfo[]) {
  const { data: metrics, connected } =
    useSSE<MetricsSnapshot>("/metrics");

  const componentMap = useMemo(() => {
    const map: Record<string, ComponentInfo> = {};
    for (const c of components) map[c.type_] = c;
    return map;
  }, [components]);

  return {
    connected,
    metrics,
    componentMap,
  };
}

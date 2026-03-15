export type NodeCategory = "source" | "conduit" | "sink";

export interface ComponentInfo {
  name: string;
  category: NodeCategory;
  init: Record<string, unknown>;
  inputs: Record<string, string>;
  outputs: Record<string, string>;
  ui_inputs: Record<string, string>;
  ui_outputs: Record<string, string>;
}

export interface GraphNode {
  type: string;
  init_args: Record<string, unknown>;
  x: number;
  y: number;
  sub_graph?: Graph;
}

export interface GraphEdge {
  source_node: string;
  source_slot: string;
  target_node: string;
  target_slot: string;
}

export interface Graph {
  nodes: Record<string, GraphNode>;
  edges: GraphEdge[];
}

export interface SenderMetrics {
  name: string;
  msg_count_delta: number;
  byte_count_delta: number;
  last_send_time: number;
  buffer_depth: number;
}

export interface ReceiverMetrics {
  name: string;
  msg_count_delta: number;
  byte_count_delta: number;
  lag: number;
}

export interface NodeMetrics {
  name: string;
  status: string;
  senders: Record<string, SenderMetrics>;
  receivers: Record<string, ReceiverMetrics>;
}

export interface MetricsSnapshot {
  nodes: Record<string, NodeMetrics>;
  timestamp: number;
}

export interface SlotType {
  name: string;
  optional: boolean;
}

export function parseSlotType(raw: string): SlotType {
  const optional = /Union\[.*,\s*NoneType\]/.test(raw) || raw.endsWith("| None");
  let name = raw
    .replace(/Union\[(.+),\s*NoneType\]/, "$1")
    .replace(/\s*\|\s*None$/, "")
    .replace(/^(?:Sender|Receiver)\[(.+)\]$/, "$1");
  // Convert Union[A, B, ...] → A | B | ...
  const unionMatch = name.match(/^Union\[(.+)\]$/);
  if (unionMatch) {
    name = unionMatch[1]!.split(/,\s*/).join(" | ");
  }
  return { name, optional };
}

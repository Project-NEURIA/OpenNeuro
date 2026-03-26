export type IOTag = "source" | "conduit" | "sink";
export type FunctionalityTag = "audio" | "video" | "llm" | "image" | "movement" | "misc" | "other";
export type GPUTag = "cpu" | "nvidia" | "apple" | "intel" | "amd";

export interface Tag {
  io: IOTag[];
  functionality: FunctionalityTag[];
  gpu: GPUTag[];
}

export type NodeCategory = IOTag | "composite";

export function categoryFromTags(tags: Tag): IOTag {
  return tags.io[0] ?? "conduit";
}

export interface ComponentInfo {
  type_: string;
  description?: string;
  tags: Tag;
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

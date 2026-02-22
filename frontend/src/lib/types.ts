export type NodeCategory = "source" | "conduit" | "sink";

export interface ComponentInfo {
  name: string;
  category: NodeCategory;
  init: Record<string, unknown>;
  inputs: Record<string, string>;
  outputs: Record<string, string>;
}

export interface GraphNode {
  type: string;
  init_args: Record<string, unknown>;
  x: number;
  y: number;
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

export interface SubscriberSnapshot {
  lag: number;
  msg_count_delta: number;
  byte_count_delta: number;
}

export interface ChannelMetrics {
  name: string;
  msg_count_delta: number;
  byte_count_delta: number;
  last_send_time: number;
  buffer_depth: number;
  subscribers: Record<string, SubscriberSnapshot>;
}

export interface NodeMetrics {
  name: string;
  status: string;
  started_at: number | null;
  channels: Record<string, ChannelMetrics>;
}

export interface MetricsSnapshot {
  nodes: Record<string, NodeMetrics>;
  timestamp: number;
}

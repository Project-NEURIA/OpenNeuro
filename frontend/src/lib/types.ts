export type NodeCategory = "source" | "conduit" | "sink";

export interface ComponentInfo {
  name: string;
  category: NodeCategory;
  init: Record<string, string>;
  inputs: Record<string, string>;
  outputs: Record<string, string>;
}

export interface ChannelMetrics {
  name: string;
  msg_count: number;
  byte_count: number;
  last_send_time: number;
  buffer_depth: number;
  subscribers: number;
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

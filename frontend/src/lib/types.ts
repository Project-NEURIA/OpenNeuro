export type NodeCategory = "source" | "conduit" | "sink";

export interface ComponentInfo {
  name: string;
  category: NodeCategory;
  inputs: string[];
  outputs: string[];
}

export interface TopicMetrics {
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
}

export interface MetricsSnapshot {
  topics: Record<string, TopicMetrics>;
  nodes: Record<string, NodeMetrics>;
  timestamp: number;
}

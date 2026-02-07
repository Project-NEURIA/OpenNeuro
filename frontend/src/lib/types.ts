export type NodeCategory = "source" | "conduit" | "sink";
export type NodeStatus = "idle" | "running" | "error" | "stopped";

export interface PipelineNode {
  id: string;
  name: string;
  category: NodeCategory;
  input_type: string | null;
  output_type: string | null;
  status: NodeStatus;
}

export interface PipelineEdge {
  id: string;
  source: string;
  target: string;
  topic_name: string;
}

export interface PipelineTopology {
  nodes: PipelineNode[];
  edges: PipelineEdge[];
}

export interface TopicMetrics {
  name: string;
  msg_count: number;
  byte_count: number;
  last_send_time: number;
  buffer_depth: number;
  subscribers: number;
  msg_per_sec: number;
}

export interface NodeMetrics {
  name: string;
  status: NodeStatus;
  started_at: number | null;
  error: string | null;
}

export interface MetricsSnapshot {
  topics: Record<string, TopicMetrics>;
  nodes: Record<string, NodeMetrics>;
  timestamp: number;
}

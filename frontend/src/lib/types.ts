export type NodeCategory = "source" | "conduit" | "sink";

export interface ComponentInfo {
  name: string;
  category: NodeCategory;
  inputs: string[];
  input_names: string[];  // Human-readable names for each input slot
  outputs: string[];
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
  channels: ChannelMetrics[];
}

export interface MetricsSnapshot {
  nodes: Record<string, NodeMetrics>;
  timestamp: number;
}

export interface FrameSnapshot {
  id: number;
  frame_type_string: string;
  pts: number;
  size_bytes: number;
  message: string; // __str__ representation
}

export interface FramesSnapshot {
  frames: FrameSnapshot[];
  timestamp: number;
}

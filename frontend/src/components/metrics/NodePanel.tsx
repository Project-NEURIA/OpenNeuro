import { cn } from "@/lib/utils";
import { Waveform } from "./Waveform";
import { ChannelSection } from "./ChannelSection";
import { formatCount, formatBytes, formatUptime } from "@/lib/format";
import type { NodeMetrics, ComponentInfo } from "@/lib/types";
import type { NodeHistoryEntry } from "@/hooks/useMetricsHistory";

interface NodePanelProps {
  nodeId: string;
  metrics: NodeMetrics;
  history: NodeHistoryEntry;
  dt: number;
  duration: number;
  componentMap: Record<string, ComponentInfo>;
  allNodes: Record<string, NodeMetrics>;
}

const categoryColors: Record<string, { border: string; badge: string; badgeColor: string }> = {
  source: {
    border: "border-source/40",
    badge: "bg-source/20",
    badgeColor: "color(display-p3 0.2 1.2 0.6)",
  },
  conduit: {
    border: "border-conduit/40",
    badge: "bg-conduit/20",
    badgeColor: "color(display-p3 0.3 0.6 1.3)",
  },
  sink: {
    border: "border-sink/40",
    badge: "bg-sink/20",
    badgeColor: "color(display-p3 1.2 0.8 0.1)",
  },
};

const statusDot: Record<string, string> = {
  running: "bg-status-running shadow-status-running/50 shadow-[0_0_6px] animate-pulse",
  startup: "bg-status-startup",
  stopped: "bg-status-stopped",
};

export function NodePanel({ nodeId, metrics, history, dt, duration, componentMap, allNodes }: NodePanelProps) {
  const info = componentMap[metrics.name];
  const category = info?.category ?? "conduit";
  const colors = categoryColors[category] ?? categoryColors.conduit!;
  const dot = statusDot[metrics.status] ?? "bg-status-stopped";
  const uptime = formatUptime(metrics.started_at);

  const startedTime = metrics.started_at
    ? new Date(metrics.started_at * 1000).toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "--";

  const currentMsg = history.msgThroughput[history.msgThroughput.length - 1] ?? 0;
  const currentBytes = history.byteThroughput[history.byteThroughput.length - 1] ?? 0;
  const channels = Object.values(metrics.channels);

  return (
    <div
      className={cn(
        "overflow-hidden",
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-white/[0.06]">
        <div className={cn("w-2.5 h-2.5 rounded-full shrink-0", dot)} />
        <span
          className="font-mono font-bold text-sm uppercase tracking-wider truncate"
          style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
        >
          {metrics.name}
        </span>
        <span className="font-mono text-[9px] text-muted-foreground/60 tabular-nums">{uptime}</span>
        <span className="font-mono text-[9px] text-muted-foreground/60 tabular-nums">{startedTime}</span>
        <span
          className={cn(
            "ml-auto text-[9px] font-bold px-2 py-0.5 rounded-md uppercase tracking-wider",
            colors.badge,
          )}
          style={{ color: colors.badgeColor }}
        >
          {category}
        </span>
        <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-muted-foreground">
          {metrics.status}
        </span>
      </div>

      {/* Node-level throughput — 2 cells */}
      <div className="grid grid-cols-2 divide-x divide-white/[0.06]">
        <div className="min-w-0 flex flex-col">
          <div className="flex items-baseline justify-between gap-2 px-3 py-1.5">
            <div className="font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
              Msg Generation Throughput
            </div>
            <div
              className="font-mono text-lg font-bold tabular-nums"
              style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
            >
              {formatCount(Math.round(currentMsg))}/s
            </div>
          </div>
          <Waveform
            data={history.msgThroughput}
            width={400}
            height={64}
            color="#4ade80"
            showAxes
            formatY={(v) => formatCount(Math.round(v))}
            duration={duration}
            className="w-full"
          />
        </div>
        <div className="min-w-0 flex flex-col">
          <div className="flex items-baseline justify-between gap-2 px-3 py-1.5">
            <div className="font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
              Byte Generation Throughput
            </div>
            <div
              className="font-mono text-lg font-bold tabular-nums"
              style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
            >
              {formatBytes(Math.round(currentBytes))}/s
            </div>
          </div>
          <Waveform
            data={history.byteThroughput}
            width={400}
            height={64}
            color="#22d3ee"
            showAxes
            formatY={(v) => formatBytes(Math.round(v))}
            duration={duration}
            className="w-full"
          />
        </div>
      </div>

      {/* Per-channel sections */}
      {channels.map((ch) => (
        <ChannelSection
          key={ch.name}
          channel={ch}
          dt={dt}
          duration={duration}
          channelHistory={history.channelHistory[ch.name]}
          nodeMap={allNodes}
        />
      ))}

      {channels.length === 0 && (
        <div className="font-mono text-[10px] text-muted-foreground/40 text-center py-2">
          No channels
        </div>
      )}
    </div>
  );
}

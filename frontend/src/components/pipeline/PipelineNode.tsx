import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import { formatCount, formatBytes, formatUptime } from "@/lib/format";
import { useVideoStream } from "@/hooks/useVideoStream";
import type { PipelineNodeData } from "@/hooks/usePipelineData";
import type { ChannelMetrics } from "@/lib/types";

const categoryColors: Record<string, { border: string; bg: string; badge: string; badgeText: string }> = {
  source: {
    border: "border-source/40",
    bg: "from-source/10 to-source/5",
    badge: "bg-source/20",
    badgeText: "text-source",
  },
  conduit: {
    border: "border-conduit/40",
    bg: "from-conduit/10 to-conduit/5",
    badge: "bg-conduit/20",
    badgeText: "text-conduit",
  },
  sink: {
    border: "border-sink/40",
    bg: "from-sink/10 to-sink/5",
    badge: "bg-sink/20",
    badgeText: "text-sink",
  },
};

const statusDot: Record<string, string> = {
  running: "bg-status-running shadow-status-running/50 shadow-[0_0_6px] animate-pulse",
  startup: "bg-status-startup",
  stopped: "bg-status-stopped",
};

/** Renders `name: Type[T]` with syntax highlighting */
function TypeLabel({ name, type }: { name: string; type: string }) {
  const match = type.match(/^(\w+)\[(.+)]$/);

  return (
    <span className="text-[12px] font-mono">
      <span style={{ color: "var(--syn-param)" }}>{name}</span>
      <span style={{ color: "var(--syn-punct)" }}>: </span>
      {match ? (
        <>
          <span style={{ color: "var(--syn-type)" }}>{match[1]}</span>
          <span style={{ color: "var(--syn-punct)" }}>[</span>
          <span style={{ color: "var(--syn-primitive)" }}>{match[2]}</span>
          <span style={{ color: "var(--syn-punct)" }}>]</span>
        </>
      ) : (
        <span style={{ color: "var(--syn-type)" }}>{type}</span>
      )}
    </span>
  );
}

function ChannelRow({ ch }: { ch: ChannelMetrics }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-foreground/60 font-medium">{ch.name}</span>
      <div className="flex items-center gap-3">
        <Stat label="msgs" value={formatCount(ch.msg_count_delta)} />
        <Stat label="bytes" value={formatBytes(ch.byte_count_delta)} />
        <Stat label="buf" value={String(ch.buffer_depth)} />
        <Stat label="subs" value={String(Object.keys(ch.subscribers).length)} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <span className="text-foreground/70 tabular-nums">{value}</span>
      <span className="text-muted-foreground/60 ml-0.5">{label}</span>
    </span>
  );
}

function PipelineNodeComponent({ id, data }: NodeProps) {
  const d = data as PipelineNodeData;
  const colors = categoryColors[d.category]!;
  const isVideoStream = d.label === "VideoStream";
  const frameUrl = useVideoStream(isVideoStream ? id : null);

  const dot = statusDot[d.status] ?? "bg-status-stopped";

  const maxSlots = Math.max(d.inputs.length, d.outputs.length, 1);

  const channels = d.nodeMetrics?.channels;
  const channelEntries = channels ? Object.values(channels) : [];
  const uptime = formatUptime(d.nodeMetrics?.started_at ?? null);

  return (
    <div
      className={cn(
        "relative rounded-2xl border px-6 py-5 min-w-[360px]",
        "bg-gradient-to-b backdrop-blur-xs",
        "bg-glass backdrop-saturate-150",
        colors.border,
        colors.bg,
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-3 pb-5 border-b border-white/[0.06]">
        <div className={cn("w-3 h-3 rounded-full shrink-0", dot)} />
        <span
          className="font-bold text-xl truncate"
          style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
        >
          {d.label}
        </span>
        <span
          className={cn(
            "ml-auto text-[11px] font-bold px-2.5 py-1 rounded-lg uppercase tracking-wider",
            colors.badge,
          )}
          style={{ color: `color(display-p3 ${d.category === "source" ? "0.2 1.2 0.6" : d.category === "sink" ? "1.2 0.8 0.1" : "0.3 0.6 1.3"})` }}
        >
          {d.category}
        </span>
      </div>

      {/* Video display for VideoStream nodes */}
      {isVideoStream && (
        <div className="py-4 border-b border-white/[0.06]">
          <div
            className="relative w-full rounded-lg overflow-hidden bg-black/60 border border-white/[0.04]"
            style={{ aspectRatio: "16/9" }}
          >
            {frameUrl ? (
              <img
                src={frameUrl}
                alt="Video stream"
                className="w-full h-full object-contain"
                draggable={false}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-muted-foreground/40 text-[11px] font-mono uppercase tracking-wider">
                  no signal
                </span>
              </div>
            )}
            {/* Scanline overlay */}
            <div
              className="pointer-events-none absolute inset-0 opacity-[0.03]"
              style={{
                backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(255,255,255,0.1) 1px, rgba(255,255,255,0.1) 2px)",
              }}
            />
          </div>
        </div>
      )}

      {/* Type rows with inline handles */}
      <div className="flex flex-col gap-3 py-5">
        {Array.from({ length: maxSlots }, (_, i) => {
          const inName = d.inputs[i];
          const outName = d.outputs[i];
          return (
            <div key={i} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {inName && (
                  <Handle
                    id={`in-${inName}`}
                    type="target"
                    position={Position.Left}
                    className="!relative !transform-none !w-4 !h-4 !bg-handle !border-handle-border !inset-auto !-ml-[32px]"
                  />
                )}
                {inName && <TypeLabel name={inName} type={d.inputTypes[inName] ?? inName} />}
              </div>
              <div className="flex items-center gap-2">
                {outName && <TypeLabel name={outName} type={d.outputTypes[outName] ?? outName} />}
                {outName && (
                  <Handle
                    id={`out-${outName}`}
                    type="source"
                    position={Position.Right}
                    className="!relative !transform-none !w-4 !h-4 !bg-handle !border-handle-border !inset-auto !-mr-[32px]"
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Metrics */}
      <div className="pt-5 border-t border-white/[0.06] text-[10px] font-mono space-y-2">
        {/* Uptime bar */}
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground/60 uppercase tracking-wider">uptime</span>
          <span className="text-foreground/70 tabular-nums">{uptime}</span>
        </div>

        {/* Channel metrics */}
        {channelEntries.length > 0 ? (
          channelEntries.map((ch) => <ChannelRow key={ch.name} ch={ch} />)
        ) : (
          <div className="text-muted-foreground/40 text-center py-1">awaiting data</div>
        )}
      </div>
    </div>
  );
}

export const PipelineNode = memo(PipelineNodeComponent);

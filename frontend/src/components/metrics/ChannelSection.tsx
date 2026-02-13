import { cn } from "@/lib/utils";
import { Waveform } from "./Waveform";
import { formatCount, formatBytes } from "@/lib/format";
import type { ChannelMetrics, NodeMetrics } from "@/lib/types";
import type { ChannelHistory } from "@/hooks/useMetricsHistory";

interface ChannelSectionProps {
  channel: ChannelMetrics;
  dt: number;
  duration: number;
  channelHistory?: ChannelHistory;
  nodeMap: Record<string, NodeMetrics>;
}

function LagBadge({ lag }: { lag: number }) {
  const color =
    lag === 0
      ? "text-status-running"
      : lag <= 5
        ? "text-amber-400"
        : "text-destructive";
  return <span className={cn("tabular-nums", color)}>{lag}</span>;
}

export function ChannelSection({ channel, dt, duration, channelHistory, nodeMap }: ChannelSectionProps) {
  const subEntries = Object.entries(channel.subscribers);
  const msgData = channelHistory?.msgThroughput ?? [];
  const byteData = channelHistory?.byteThroughput ?? [];
  const bufData = channelHistory?.bufferDepths ?? [];

  const rate = dt > 0 ? 1 / dt : 0;
  const msgPerSec = channel.msg_count_delta * rate;
  const bytesPerSec = channel.byte_count_delta * rate;

  return (
    <div className="border-t border-white/[0.06]">
      {/* Channel header */}
      <div className="flex items-center justify-between px-3 py-1.5">
        <span className="font-mono text-[12px] font-medium" style={{ color: "var(--syn-param)" }}>
          {channel.name}
        </span>
        <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.15em]">
          Subs{" "}
          <span className="text-foreground/70 tabular-nums">{subEntries.length}</span>
        </span>
      </div>

      {/* 3 metric cells */}
      <div className="grid grid-cols-3 border-t border-white/[0.06] divide-x divide-white/[0.06]">
        <div className="min-w-0 flex flex-col">
          <div className="flex items-baseline justify-between gap-1 px-2 py-1">
            <div className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
              Msg Write Thru
            </div>
            <div
              className="font-mono text-sm font-bold tabular-nums"
              style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
            >
              {formatCount(Math.round(msgPerSec))}/s
            </div>
          </div>
          <Waveform
            data={msgData}
            width={300}
            height={48}
            color="#4ade80"
            showAxes
            formatY={(v) => formatCount(Math.round(v))}
            duration={duration}
            className="w-full"
          />
        </div>
        <div className="min-w-0 flex flex-col">
          <div className="flex items-baseline justify-between gap-1 px-2 py-1">
            <div className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
              Byte Write Thru
            </div>
            <div
              className="font-mono text-sm font-bold tabular-nums"
              style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
            >
              {formatBytes(Math.round(bytesPerSec))}/s
            </div>
          </div>
          <Waveform
            data={byteData}
            width={300}
            height={48}
            color="#22d3ee"
            showAxes
            formatY={(v) => formatBytes(Math.round(v))}
            duration={duration}
            className="w-full"
          />
        </div>
        <div className="min-w-0 flex flex-col">
          <div className="flex items-baseline justify-between gap-1 px-2 py-1">
            <div className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
              Buffer Size
            </div>
            <div
              className="font-mono text-sm font-bold tabular-nums"
              style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
            >
              {channel.buffer_depth}
            </div>
          </div>
          <Waveform
            data={bufData}
            width={300}
            height={48}
            color="#fbbf24"
            showAxes
            duration={duration}
            className="w-full"
          />
        </div>
      </div>

      {/* Subscribers */}
      {subEntries.map(([subId, sub]) => {
        const name = nodeMap[subId]?.name ?? subId;
        const subMsgPerSec = sub.msg_count_delta * rate;
        const subBytesPerSec = sub.byte_count_delta * rate;
        const subHist = channelHistory?.subscriberHistory[subId];
        const subMsgData = subHist?.msgThroughput ?? [];
        const subByteData = subHist?.byteThroughput ?? [];
        return (
          <div key={subId} className="border-t border-white/[0.06] flex">
            {/* Left: name + lag */}
            <div className="flex flex-col justify-center gap-1 px-3 py-1.5 border-r border-white/[0.06] min-w-[100px]">
              <div className="flex items-center gap-1.5 font-mono text-[11px]">
                <span className="text-muted-foreground">→</span>
                <span className="text-foreground/80 font-medium">{name}</span>
              </div>
              <div className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.15em]">
                lag <LagBadge lag={sub.lag} />
              </div>
            </div>
            {/* Right: two graph cells */}
            <div className="flex-1 grid grid-cols-2 divide-x divide-white/[0.06] min-w-0">
              <div className="min-w-0 flex flex-col">
                <div className="flex items-baseline justify-between gap-1 px-2 py-1">
                  <div className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
                    Msg Read Thru
                  </div>
                  <div
                    className="font-mono text-sm font-bold tabular-nums"
                    style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
                  >
                    {formatCount(Math.round(subMsgPerSec))}/s
                  </div>
                </div>
                <Waveform
                  data={subMsgData}
                  width={300}
                  height={48}
                  color="#4ade80"
                  showAxes
                  formatY={(v) => formatCount(Math.round(v))}
                  duration={duration}
                  className="w-full"
                />
              </div>
              <div className="min-w-0 flex flex-col">
                <div className="flex items-baseline justify-between gap-1 px-2 py-1">
                  <div className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
                    Byte Read Thru
                  </div>
                  <div
                    className="font-mono text-sm font-bold tabular-nums"
                    style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
                  >
                    {formatBytes(Math.round(subBytesPerSec))}/s
                  </div>
                </div>
                <Waveform
                  data={subByteData}
                  width={300}
                  height={48}
                  color="#22d3ee"
                  showAxes
                  formatY={(v) => formatBytes(Math.round(v))}
                  duration={duration}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

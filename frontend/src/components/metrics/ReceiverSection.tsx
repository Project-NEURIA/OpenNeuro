import { cn } from "@/lib/utils";
import { Waveform } from "./Waveform";
import { formatCount, formatBytes } from "@/lib/format";
import type { ReceiverMetrics } from "@/lib/types";
import type { ReceiverHistory } from "@/hooks/useMetricsHistory";

interface ReceiverSectionProps {
  receiver: ReceiverMetrics;
  dt: number;
  duration: number;
  receiverHistory?: ReceiverHistory;
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

export function ReceiverSection({ receiver, dt, duration, receiverHistory }: ReceiverSectionProps) {
  const msgData = receiverHistory?.msgThroughput ?? [];
  const byteData = receiverHistory?.byteThroughput ?? [];

  const rate = dt > 0 ? 1 / dt : 0;
  const msgPerSec = receiver.msg_count_delta * rate;
  const bytesPerSec = receiver.byte_count_delta * rate;

  return (
    <div className="border-t border-white/[0.06]">
      {/* Receiver header */}
      <div className="flex items-center justify-between px-3 py-1.5">
        <div className="flex items-center gap-1.5 font-mono text-[12px]">
          <span className="text-muted-foreground">→</span>
          <span className="font-medium" style={{ color: "var(--syn-param)" }}>
            {receiver.name}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.15em]">
            lag <LagBadge lag={receiver.lag} />
          </span>
          <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.15em]">
            Input
          </span>
        </div>
      </div>

      {/* 2 metric cells */}
      <div className="grid grid-cols-2 border-t border-white/[0.06] divide-x divide-white/[0.06]">
        <div className="min-w-0 flex flex-col">
          <div className="flex items-baseline justify-between gap-1 px-2 py-1">
            <div className="font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
              Msg Read Thru
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
              Byte Read Thru
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
      </div>
    </div>
  );
}

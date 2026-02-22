import { Waveform } from "./Waveform";
import { formatCount, formatBytes } from "@/lib/format";
import type { SenderMetrics } from "@/lib/types";
import type { SenderHistory } from "@/hooks/useMetricsHistory";

interface SenderSectionProps {
  sender: SenderMetrics;
  dt: number;
  duration: number;
  senderHistory?: SenderHistory;
}

export function SenderSection({ sender, dt, duration, senderHistory }: SenderSectionProps) {
  const msgData = senderHistory?.msgThroughput ?? [];
  const byteData = senderHistory?.byteThroughput ?? [];
  const bufData = senderHistory?.bufferDepths ?? [];

  const rate = dt > 0 ? 1 / dt : 0;
  const msgPerSec = sender.msg_count_delta * rate;
  const bytesPerSec = sender.byte_count_delta * rate;

  return (
    <div className="border-t border-white/[0.06]">
      {/* Sender header */}
      <div className="flex items-center justify-between px-3 py-1.5">
        <span className="font-mono text-[12px] font-medium" style={{ color: "var(--syn-param)" }}>
          {sender.name}
        </span>
        <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.15em]">
          Output
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
              {sender.buffer_depth}
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
    </div>
  );
}

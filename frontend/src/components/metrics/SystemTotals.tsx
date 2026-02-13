import { Waveform } from "./Waveform";
import { formatCount, formatBytes } from "@/lib/format";
import type { MetricsHistory } from "@/hooks/useMetricsHistory";

interface SystemTotalsProps {
  history: MetricsHistory;
}

function TotalCell({
  label,
  value,
  waveData,
  color,
  formatY,
  duration,
}: {
  label: string;
  value: string;
  waveData: number[];
  color: string;
  formatY?: (v: number) => string;
  duration: number;
}) {
  return (
    <div className="min-w-0 flex flex-col">
      <div className="flex items-baseline justify-between gap-2 px-3 py-1.5">
        <div className="font-mono text-[10px] uppercase tracking-[0.15em] text-muted-foreground whitespace-nowrap">
          {label}
        </div>
        <div
          className="font-mono text-lg font-bold tabular-nums"
          style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
        >
          {value}
        </div>
      </div>
      <Waveform
        data={waveData}
        width={400}
        height={120}
        color={color}
        showAxes
        formatY={formatY}
        duration={duration}
        className="w-full"
      />
    </div>
  );
}

export function SystemTotals({ history }: SystemTotalsProps) {
  const { current, nodeHistory, snapshots } = history;

  // Compute window duration from snapshot timestamps
  const first = snapshots[0];
  const last = snapshots[snapshots.length - 1];
  const duration = first && last ? last.timestamp - first.timestamp : 0;

  // Aggregate per-second throughput from node histories
  const len = Object.values(nodeHistory)[0]?.msgThroughput.length ?? 0;
  const totalMsgThroughput: number[] = new Array(len).fill(0);
  const totalByteThroughput: number[] = new Array(len).fill(0);

  for (const nh of Object.values(nodeHistory)) {
    for (let i = 0; i < len; i++) {
      totalMsgThroughput[i]! += nh.msgThroughput[i] ?? 0;
      totalByteThroughput[i]! += nh.byteThroughput[i] ?? 0;
    }
  }

  const currentMsg = totalMsgThroughput[totalMsgThroughput.length - 1] ?? 0;
  const currentBytes = totalByteThroughput[totalByteThroughput.length - 1] ?? 0;

  return (
    <div className="grid grid-cols-2 border-b border-white/[0.06] divide-x divide-white/[0.06]">
      <TotalCell
        label="Total Msg Throughput"
        value={`${formatCount(Math.round(currentMsg))}/s`}
        waveData={totalMsgThroughput}
        color="#4ade80"
        formatY={(v) => formatCount(Math.round(v))}
        duration={duration}
      />
      <TotalCell
        label="Total Byte Throughput"
        value={`${formatBytes(Math.round(currentBytes))}/s`}
        waveData={totalByteThroughput}
        color="#22d3ee"
        formatY={(v) => formatBytes(Math.round(v))}
        duration={duration}
      />
    </div>
  );
}

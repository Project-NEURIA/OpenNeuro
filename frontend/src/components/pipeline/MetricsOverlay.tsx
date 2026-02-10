import { Play, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { startAll, stopAll } from "@/lib/api";
import type { MetricsSnapshot } from "@/lib/types";

interface MetricsOverlayProps {
  connected: boolean;
  metrics: MetricsSnapshot | null;
}

export function MetricsOverlay({ connected, metrics }: MetricsOverlayProps) {
  const totalMsgs = metrics
    ? Object.values(metrics.nodes).reduce(
        (s, n) => s + n.channels.reduce((s2, t) => s2 + t.msg_count, 0),
        0,
      )
    : 0;

  const nodeCount = metrics ? Object.keys(metrics.nodes).length : 0;
  const runningCount = metrics
    ? Object.values(metrics.nodes).filter((n) => n.status === "running").length
    : 0;

  const isRunning = runningCount > 0;

  return (
    <div className="absolute top-3 right-3 z-10 flex items-center gap-3 bg-zinc-900/80 backdrop-blur-sm border border-zinc-800 rounded-lg px-3 py-1.5 text-[11px]">
      <div className="flex items-center gap-1.5">
        <div
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            connected
              ? "bg-green-400 shadow-green-400/50 shadow-[0_0_4px]"
              : "bg-red-500",
          )}
        />
        <span className="text-zinc-400">
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      <div className="w-px h-3 bg-zinc-700" />

      <span className="text-zinc-500">
        Nodes:{" "}
        <span className="text-zinc-300 font-mono">
          {runningCount}/{nodeCount}
        </span>
      </span>

      <div className="w-px h-3 bg-zinc-700" />

      <span className="text-zinc-500">
        Total:{" "}
        <span className="text-zinc-300 font-mono">
          {totalMsgs.toLocaleString()}
        </span>
      </span>

      <div className="w-px h-3 bg-zinc-700" />

      <button
        onClick={() => startAll().catch(console.error)}
        disabled={isRunning}
        className={cn(
          "p-1 rounded transition-colors",
          isRunning
            ? "text-zinc-600 cursor-not-allowed"
            : "text-emerald-400 hover:bg-zinc-800",
        )}
      >
        <Play className="w-3.5 h-3.5" />
      </button>
      <button
        onClick={() => stopAll().catch(console.error)}
        disabled={!isRunning}
        className={cn(
          "p-1 rounded transition-colors",
          !isRunning
            ? "text-zinc-600 cursor-not-allowed"
            : "text-red-400 hover:bg-zinc-800",
        )}
      >
        <Square className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

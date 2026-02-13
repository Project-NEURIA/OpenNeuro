import { Play, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { startAll, stopAll } from "@/lib/api";
import type { MetricsSnapshot } from "@/lib/types";

interface MetricsOverlayProps {
  connected: boolean;
  metrics: MetricsSnapshot | null;
}

export function MetricsOverlay({ connected, metrics }: MetricsOverlayProps) {
  const nodeCount = metrics ? Object.keys(metrics.nodes).length : 0;
  const runningCount = metrics
    ? Object.values(metrics.nodes).filter((n) => n.status === "running").length
    : 0;

  const isRunning = runningCount > 0;

  return (
    <div className="absolute top-4 right-4 z-10 flex items-center gap-3 bg-glass backdrop-blur-xs backdrop-saturate-150 border border-glass-border shadow-2xl shadow-black/40 rounded-2xl px-4 py-2 text-[11px]">
      <div className="flex items-center gap-1.5">
        <div
          className={cn(
            "w-1.5 h-1.5 rounded-full",
            connected
              ? "bg-status-running shadow-status-running/50 shadow-[0_0_4px]"
              : "bg-destructive",
          )}
        />
        <span className="text-muted-foreground">
          {connected ? "Live" : "Disconnected"}
        </span>
      </div>

      <div className="w-px h-3 bg-border" />

      <span className="text-muted-foreground">
        Nodes:{" "}
        <span className="text-foreground font-mono">
          {runningCount}/{nodeCount}
        </span>
      </span>

      <div className="w-px h-3 bg-border" />

      <button
        onClick={() => startAll().catch(console.error)}
        disabled={isRunning}
        className={cn(
          "p-1 rounded transition-colors",
          isRunning
            ? "text-muted-foreground/50 cursor-not-allowed"
            : "text-status-running hover:bg-accent",
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
            ? "text-muted-foreground/50 cursor-not-allowed"
            : "text-destructive hover:bg-accent",
        )}
      >
        <Square className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

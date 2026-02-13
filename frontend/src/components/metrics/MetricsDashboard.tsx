import { useMemo } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { SystemTotals } from "./SystemTotals";
import { NodePanel } from "./NodePanel";
import type { MetricsHistory } from "@/hooks/useMetricsHistory";
import type { ComponentInfo } from "@/lib/types";

interface MetricsDashboardProps {
  connected: boolean;
  history: MetricsHistory;
  componentMap: Record<string, ComponentInfo>;
  onClose: () => void;
}

const CATEGORY_ORDER: Record<string, number> = { source: 0, conduit: 1, sink: 2 };

export function MetricsDashboard({ connected, history, componentMap, onClose }: MetricsDashboardProps) {
  const { current, nodeHistory, dt, snapshots } = history;
  const nodes = current?.nodes ?? {};

  const firstSnap = snapshots[0];
  const lastSnap = snapshots[snapshots.length - 1];
  const duration = firstSnap && lastSnap ? lastSnap.timestamp - firstSnap.timestamp : 0;

  const sortedNodeIds = useMemo(() => {
    return Object.keys(nodes).sort((a, b) => {
      const na = nodes[a]!;
      const nb = nodes[b]!;
      const catA = componentMap[na.name]?.category ?? "conduit";
      const catB = componentMap[nb.name]?.category ?? "conduit";
      const orderDiff = (CATEGORY_ORDER[catA] ?? 1) - (CATEGORY_ORDER[catB] ?? 1);
      if (orderDiff !== 0) return orderDiff;
      return na.name.localeCompare(nb.name);
    });
  }, [nodes, componentMap]);

  return (
    <div
      className={cn(
        "absolute top-0 right-0 z-20 w-1/2 h-full",
        "border-l border-glass-border",
        "bg-glass backdrop-blur-xs backdrop-saturate-150",
        "shadow-2xl shadow-black/40",
        "flex flex-col",
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-white">Metrics</h2>
          <div
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              connected
                ? "bg-status-running shadow-status-running/50 shadow-[0_0_4px]"
                : "bg-destructive",
            )}
          />
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Totals */}
      <SystemTotals history={history} />

      {/* Scrollable node list */}
      <div className="flex-1 overflow-y-auto divide-y divide-white/[0.06]">
        {sortedNodeIds.map((nodeId) => (
          <NodePanel
            key={nodeId}
            nodeId={nodeId}
            metrics={nodes[nodeId]!}
            history={nodeHistory[nodeId] ?? { msgThroughput: [], byteThroughput: [], channelHistory: {} }}
            dt={dt}
            duration={duration}
            componentMap={componentMap}
            allNodes={nodes}
          />
        ))}

        {sortedNodeIds.length === 0 && (
          <div className="flex items-center justify-center h-32 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
            Awaiting pipeline data
          </div>
        )}
      </div>
    </div>
  );
}

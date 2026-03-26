import { useMemo } from "react";
import type { Node } from "@xyflow/react";
import type { GraphNodeData } from "@/hooks/useGraphData";
import { useComponentLogs } from "@/hooks/useComponentLogs";
import { cn } from "@/lib/utils";

interface LoggingPanelProps {
  selectedNode: Node | null;
}

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString();
}

export function LoggingPanel({ selectedNode }: LoggingPanelProps) {
  const nodeId = selectedNode?.id ?? null;
  const { entries, error } = useComponentLogs(nodeId);
  const data = selectedNode?.data as GraphNodeData | undefined;

  const stdoutCount = useMemo(
    () => entries.filter((entry) => entry.stream === "stdout").length,
    [entries],
  );
  const stderrCount = useMemo(
    () => entries.filter((entry) => entry.stream === "stderr").length,
    [entries],
  );

  if (!selectedNode || !data) {
    return (
      <div className="absolute top-20 right-4 z-10 w-[520px] max-w-[calc(100vw-2rem)] rounded-xl border border-glass-border bg-glass/95 p-3 text-xs backdrop-blur-md">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          Logging
        </div>
        <div className="mt-2 text-muted-foreground">No component selected.</div>
      </div>
    );
  }

  return (
    <div className="absolute top-20 right-4 z-10 w-[520px] max-w-[calc(100vw-2rem)] max-h-[calc(100vh-6rem)] overflow-hidden rounded-xl border border-glass-border bg-glass/95 p-3 text-xs backdrop-blur-md">
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          Logging
        </div>
        <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
          stdout {stdoutCount} | stderr {stderrCount}
        </div>
      </div>

      <div className="mt-2 rounded-md border border-white/10 bg-black/30 p-2">
        <div className="flex items-center justify-between gap-2">
          <div className="truncate font-mono text-sm text-foreground">{data.label}</div>
          <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
            {data.status}
          </div>
        </div>
        <div className="mt-1 text-[11px] text-muted-foreground">
          Showing captured process stdout/stderr for this component thread.
        </div>
      </div>

      <div className="mt-2 h-[360px] overflow-auto rounded-md border border-white/10 bg-black/35 p-2 font-mono text-[11px]">
        {entries.length === 0 && (
          <div className="text-muted-foreground">
            No logs captured yet for this component.
          </div>
        )}
        {entries.map((entry) => (
          <div key={entry.seq} className="mb-1 break-words">
            <span className="text-muted-foreground/80">[{formatTimestamp(entry.timestamp)}]</span>{" "}
            <span
              className={cn(
                "uppercase",
                entry.stream === "stderr" ? "text-status-startup" : "text-status-running",
              )}
            >
              {entry.stream}
            </span>{" "}
            <span className="text-foreground/95">{entry.text}</span>
          </div>
        ))}
        {error && (
          <div className="mt-2 text-status-startup">
            Failed to load logs: {error}
          </div>
        )}
      </div>
    </div>
  );
}

import { useState } from "react";
import type { Node } from "@xyflow/react";
import type { GraphNodeData } from "@/hooks/useGraphData";
import type { MetricsSnapshot } from "@/lib/types";
import { cn } from "@/lib/utils";

interface LoggingPanelProps {
  selectedNode: Node | null;
  metrics: MetricsSnapshot | null;
}

function formatEpochSeconds(ts: number | null): string {
  if (!ts || !Number.isFinite(ts)) return "--";
  return new Date(ts * 1000).toLocaleString();
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "--";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(2)}s`;
  const mins = Math.floor(seconds / 60);
  const rem = seconds % 60;
  return `${mins}m ${rem.toFixed(1)}s`;
}

function getSendStatus(maxLastSend: number, dtSinceLastSend: number | null) {
  if (maxLastSend <= 0) {
    return { label: "No send data yet", detail: "" };
  }
  if (dtSinceLastSend === null) {
    return { label: "Unable to determine send freshness", detail: "Different time bases" };
  }
  if (dtSinceLastSend < 0.5) {
    return { label: "Send activity just now", detail: `Within ${formatDuration(dtSinceLastSend)}` };
  }
  if (dtSinceLastSend <= 2) {
    return { label: "Recent send activity", detail: `${formatDuration(dtSinceLastSend)} ago` };
  }
  return { label: "No send activity for a while", detail: `No sends for ${formatDuration(dtSinceLastSend)}` };
}

export function LoggingPanel({ selectedNode, metrics }: LoggingPanelProps) {
  const [showRaw, setShowRaw] = useState(false);

  if (!selectedNode) {
    return (
      <div className="absolute top-20 right-4 z-10 w-[440px] max-w-[calc(100vw-2rem)] rounded-xl border border-glass-border bg-glass/95 p-3 text-xs backdrop-blur-md">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          Logging
        </div>
        <div className="mt-2 text-muted-foreground">
          No component selected.
        </div>
      </div>
    );
  }

  const data = selectedNode.data as GraphNodeData;
  const nodeMetrics = data.nodeMetrics ?? metrics?.nodes[selectedNode.id] ?? null;
  const senders = nodeMetrics ? Object.values(nodeMetrics.senders) : [];
  const receivers = nodeMetrics ? Object.values(nodeMetrics.receivers) : [];
  const totalMsgDelta = senders.reduce((sum, s) => sum + (s.msg_count_delta ?? 0), 0);
  const totalByteDelta = senders.reduce((sum, s) => sum + (s.byte_count_delta ?? 0), 0);
  const maxBufferDepth = senders.reduce((max, s) => Math.max(max, s.buffer_depth ?? 0), 0);
  const maxLastSend = senders.reduce((max, s) => Math.max(max, s.last_send_time ?? 0), 0);
  const metricsTimestamp = metrics?.timestamp ?? null;
  const hasComparableEpoch = metricsTimestamp !== null && maxLastSend > 0 && metricsTimestamp > 1e9 && maxLastSend > 1e9;
  const hasComparableRelative = metricsTimestamp !== null && maxLastSend > 0 && metricsTimestamp > 0 && metricsTimestamp < 1e9 && maxLastSend < 1e9;
  const canCompareLastSend = hasComparableEpoch || hasComparableRelative;
  const dtSinceLastSend = canCompareLastSend ? Math.max(0, (metricsTimestamp as number) - maxLastSend) : null;
  const sendStatus = getSendStatus(maxLastSend, dtSinceLastSend);

  const warnings: string[] = [];
  if (maxBufferDepth > 1000) {
    warnings.push(`Buffer backlog is high (${maxBufferDepth}).`);
  }
  if (nodeMetrics?.status === "running" && totalMsgDelta === 0 && receivers.length === 0) {
    warnings.push("Running but no outgoing messages detected in this tick.");
  }
  if (nodeMetrics?.status === "running" && dtSinceLastSend !== null && dtSinceLastSend > 2) {
    warnings.push(`No send activity for ${dtSinceLastSend.toFixed(2)}s.`);
  }
  const health = warnings.length === 0
    ? { level: "ok" as const, lines: ["Node appears healthy."] }
    : { level: "warn" as const, lines: warnings };

  const snapshot = {
    selectedNodeId: selectedNode.id,
    component: data.label,
    category: data.category,
    status: data.status,
    position: selectedNode.position,
    initArgs: data.initArgs ?? {},
    inputs: data.inputs,
    outputs: data.outputs,
    inputTypes: data.inputTypes,
    outputTypes: data.outputTypes,
    resolvedTypes: data.resolvedTypes ?? {},
    ui_inputs: data.ui_inputs,
    ui_outputs: data.ui_outputs,
    nodeMetrics,
    metricsTimestamp,
    metricsTimestampFormatted: formatEpochSeconds(metricsTimestamp),
    lastSendTime: maxLastSend || null,
    lastSendComparable: canCompareLastSend,
    lastSendAgo: dtSinceLastSend,
  };

  return (
    <div className="absolute top-20 right-4 z-10 w-[440px] max-w-[calc(100vw-2rem)] max-h-[calc(100vh-6rem)] overflow-auto rounded-xl border border-glass-border bg-glass/95 p-3 text-xs backdrop-blur-md">
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
          Logging
        </div>
        <div
          className={cn(
            "shrink-0 rounded px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider",
            health.level === "ok" ? "bg-status-running/15 text-status-running" : "bg-status-startup/15 text-status-startup",
          )}
        >
          {health.level === "ok" ? "Healthy" : "Warning"}
        </div>
      </div>

      <div className="mt-2 rounded-md border border-white/10 bg-black/30 p-2">
        <div className="flex items-center justify-between gap-2">
          <div className="truncate font-mono text-sm text-foreground">{data.label}</div>
          <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{nodeMetrics?.status ?? data.status}</div>
        </div>
        <div className="mt-2 grid grid-cols-3 gap-2">
          <div className="rounded border border-white/10 bg-black/25 p-2">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Throughput</div>
            <div className="mt-1 font-mono text-foreground">{totalMsgDelta} msg/tick</div>
          </div>
          <div className="rounded border border-white/10 bg-black/25 p-2">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Bytes</div>
            <div className="mt-1 font-mono text-foreground">{totalByteDelta} B/tick</div>
          </div>
          <div className="rounded border border-white/10 bg-black/25 p-2">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Max Buffer</div>
            <div className={cn("mt-1 font-mono", maxBufferDepth > 1000 ? "text-status-startup" : "text-foreground")}>{maxBufferDepth}</div>
          </div>
        </div>
        <div className="mt-2 text-[11px] text-muted-foreground">
          Metrics time:{" "}
          <span className="font-mono text-foreground">
            {metricsTimestamp ? formatEpochSeconds(metricsTimestamp) : "--"}
          </span>
        </div>
        <div className="mt-1 text-[11px] text-muted-foreground">
          Send status:{" "}
          <span className="font-mono text-foreground">{sendStatus.label}</span>
        </div>
        {sendStatus.detail && (
          <div className="mt-1 text-[11px] text-muted-foreground/80">
            {sendStatus.detail}
          </div>
        )}
      </div>

      <div className="mt-2 rounded-md border border-white/10 bg-black/30 p-2">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">Health Analysis</div>
        <div className="mt-1 space-y-1">
          {health.lines.map((line) => (
            <div key={line} className={cn("text-[11px]", health.level === "ok" ? "text-status-running" : "text-status-startup")}>
              {health.level === "ok" ? "OK: " : "WARN: "}
              {line}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-2 rounded-md border border-white/10 bg-black/30 p-2">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">I/O Metrics</div>
        <div className="mt-2 space-y-1">
          {senders.map((s) => (
            <div key={`sender-${s.name}`} className="rounded border border-white/10 bg-black/20 px-2 py-1">
              <div className="font-mono text-[11px] text-foreground">out: {s.name}</div>
              <div className="font-mono text-[10px] text-muted-foreground">
                msg +{s.msg_count_delta} | bytes +{s.byte_count_delta} | buffer {s.buffer_depth}
              </div>
            </div>
          ))}
          {receivers.map((r) => (
            <div key={`receiver-${r.name}`} className="rounded border border-white/10 bg-black/20 px-2 py-1">
              <div className="font-mono text-[11px] text-foreground">in: {r.name}</div>
              <div className="font-mono text-[10px] text-muted-foreground">
                msg +{r.msg_count_delta} | bytes +{r.byte_count_delta} | lag {r.lag}
              </div>
            </div>
          ))}
          {senders.length === 0 && receivers.length === 0 && (
            <div className="text-[11px] text-muted-foreground">No channel metrics.</div>
          )}
        </div>
      </div>

      <div className="mt-2 rounded-md border border-white/10 bg-black/30 p-2">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">Config</div>
        <pre className="mt-1 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] text-foreground/90">
          {JSON.stringify(data.initArgs ?? {}, null, 2)}
        </pre>
      </div>

      <div className="mt-2 rounded-md border border-white/10 bg-black/30 p-2">
        <button
          onClick={() => setShowRaw((v) => !v)}
          className="rounded border border-white/10 bg-black/20 px-2 py-1 font-mono text-[11px] text-muted-foreground hover:text-foreground"
        >
          {showRaw ? "Hide Raw JSON" : "View Raw JSON"}
        </button>
        {showRaw && (
          <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-[11px] text-foreground/90">
            {JSON.stringify(snapshot, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

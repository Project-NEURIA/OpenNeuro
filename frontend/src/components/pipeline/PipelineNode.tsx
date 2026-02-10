import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { PipelineNodeData } from "@/hooks/usePipelineData";

const categoryColors: Record<string, { border: string; bg: string; badge: string; badgeText: string }> = {
  source: {
    border: "border-emerald-500/40",
    bg: "from-emerald-500/10 to-emerald-500/5",
    badge: "bg-emerald-500/20",
    badgeText: "text-emerald-400",
  },
  conduit: {
    border: "border-blue-500/40",
    bg: "from-blue-500/10 to-blue-500/5",
    badge: "bg-blue-500/20",
    badgeText: "text-blue-400",
  },
  sink: {
    border: "border-amber-500/40",
    bg: "from-amber-500/10 to-amber-500/5",
    badge: "bg-amber-500/20",
    badgeText: "text-amber-400",
  },
};

const statusDot: Record<string, string> = {
  running: "bg-green-400 shadow-green-400/50 shadow-[0_0_6px] animate-pulse",
  startup: "bg-yellow-400",
  stopped: "bg-zinc-500",
};

function PipelineNodeComponent({ data }: NodeProps) {
  const d = data as PipelineNodeData;
  const colors = categoryColors[d.category]!;
  const dot = statusDot[d.status] ?? "bg-zinc-500";

  return (
    <div
      className={cn(
        "relative rounded-xl border px-4 py-3 min-w-[180px]",
        "bg-gradient-to-b backdrop-blur-md",
        "bg-zinc-900/80",
        colors.border,
        colors.bg,
      )}
    >
      {/* Input handles */}
      {d.inputs.map((type, i) => (
        <Handle
          key={`in-${i}`}
          id={`in-${i}`}
          type="target"
          position={Position.Left}
          className="!w-3 !h-3 !bg-zinc-600 !border-zinc-500"
          style={{ top: `${((i + 1) / (d.inputs.length + 1)) * 100}%` }}
        />
      ))}

      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <div className={cn("w-2 h-2 rounded-full shrink-0", dot)} />
        <span className="font-semibold text-sm text-zinc-100 truncate">
          {d.label}
        </span>
        <span
          className={cn(
            "ml-auto text-[10px] font-medium px-1.5 py-0.5 rounded-md uppercase tracking-wider",
            colors.badge,
            colors.badgeText,
          )}
        >
          {d.category}
        </span>
      </div>

      {/* Type labels */}
      <div className="flex items-center justify-between text-[10px] text-zinc-500 mb-2">
        {d.inputs.length > 0 && (
          <span className="font-mono">{d.inputs.join(", ")}</span>
        )}
        {d.inputs.length > 0 && d.outputs.length > 0 && <span>-&gt;</span>}
        {d.outputs.length > 0 && (
          <span className={cn("font-mono", d.inputs.length === 0 && "ml-auto")}>
            {d.outputs.join(", ")}
          </span>
        )}
      </div>

      {/* Live metrics */}
      {d.nodeMetrics && d.nodeMetrics.channels.length > 0 && (
        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[10px] border-t border-zinc-700/50 pt-1.5">
          <div className="text-zinc-500">total</div>
          <div className="text-zinc-500">buf</div>
          <div className="text-zinc-300 font-mono">
            {d.nodeMetrics.channels.reduce((s, t) => s + t.msg_count, 0).toLocaleString()}
          </div>
          <div className="text-zinc-300 font-mono">
            {d.nodeMetrics.channels.reduce((s, t) => s + t.buffer_depth, 0)}
          </div>
        </div>
      )}

      {/* Output handles */}
      {d.outputs.map((type, i) => (
        <Handle
          key={`out-${i}`}
          id={`out-${i}`}
          type="source"
          position={Position.Right}
          className="!w-3 !h-3 !bg-zinc-600 !border-zinc-500"
          style={{ top: `${((i + 1) / (d.outputs.length + 1)) * 100}%` }}
        />
      ))}
    </div>
  );
}

export const PipelineNode = memo(PipelineNodeComponent);

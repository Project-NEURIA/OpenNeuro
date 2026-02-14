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

  // Calculate minimum height based on content
  const baseHeight = 160; // Base minimum height
  const inputHeight = d.inputs.length > 0 ? (20 + (d.inputs.length * 40)) : baseHeight; // Last input at 20 + (n-1)*40
  const outputHeight = d.outputs.length > 0 ? (20 + (d.outputs.length * 40)) : baseHeight; // Last output at 20 + (n-1)*40
  const minHeight = Math.max(baseHeight, inputHeight, outputHeight);

  return (
    <div
      className={cn(
        "relative rounded-xl border px-4 py-3 flex flex-col",
        "bg-gradient-to-b backdrop-blur-md",
        "bg-zinc-900/80",
        colors.border,
        colors.bg,
      )}
      style={{ 
        minHeight: `${minHeight}px`,
        height: `${minHeight}px`,
        width: '200px' // Match App.tsx
      }}
      data-width={200}
      data-height={minHeight}
    >
      {/* Input handles with labels */}
      {d.inputs.map((type, i) => (
        <div
          key={`in-${i}`}
          className="absolute flex items-center gap-1"
          style={{
            left: 0,
            top: `${60 + (i * 40)}px`, // Start after header (40px) + margin (20px)
            transform: 'translateY(-50%)',
          }}
        >
          <Handle
            id={`in-${i}`}
            type="target"
            position={Position.Left}
            className="!w-3 !h-3 !bg-zinc-600 !border-zinc-500 !relative !left-0"
          />
          <span className="text-[9px] text-zinc-500 font-medium whitespace-nowrap">
            {d.input_names?.[i] || `input_${i}`} ({d.inputs[i]})
          </span>
        </div>
      ))}

      {/* Header */}
      <div className="flex items-center gap-2 mb-2 flex-shrink-0">
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

      {/* Live metrics - make this expand to fill remaining space */}
      <div className="flex-1 flex items-end">
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
      </div>

      {/* Output handles */}
      {d.outputs.map((type, i) => (
        <div
          key={`out-${i}`}
          className="absolute flex items-center gap-1"
          style={{
            right: 0,
            top: `${60 + (i * 40)}px`, // Start after header (40px) + margin (20px)
            transform: 'translateY(-50%)',
          }}
        >
          <Handle
            id={`out-${i}`}
            type="source"
            position={Position.Right}
            className="!w-3 !h-3 !bg-zinc-600 !border-zinc-500"
          />
          <span className="text-[8px] text-zinc-400 ml-2 px-2 py-1">
            ({d.outputs[i]})
          </span>
        </div>
      ))}
    </div>
  );
}

export const PipelineNode = memo(PipelineNodeComponent);

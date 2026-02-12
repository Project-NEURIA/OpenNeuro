import { Mic, AudioLines, MessageSquareText, Brain, Volume2, Radio, Speaker, Puzzle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ComponentInfo } from "@/lib/types";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  Mic,
  VAD: AudioLines,
  ASR: MessageSquareText,
  LLM: Brain,
  TTS: Volume2,
  STS: Radio,
  Speaker,
};

const catColor: Record<string, string> = {
  source: "border-emerald-500/30 hover:border-emerald-500/60",
  conduit: "border-blue-500/30 hover:border-blue-500/60",
  sink: "border-amber-500/30 hover:border-amber-500/60",
};

interface NodeSidebarProps {
  components: ComponentInfo[];
}

export function NodeSidebar({ components }: NodeSidebarProps) {
  function onDragStart(e: React.DragEvent, item: ComponentInfo) {
    e.dataTransfer.setData("application/pipeline-node", JSON.stringify(item));
    e.dataTransfer.effectAllowed = "move";
  }

  return (
    <div className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-950 p-4 flex flex-col gap-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
        Components
      </h2>
      {components.map((item) => {
        const Icon = iconMap[item.name] ?? Puzzle;
        return (
          <div
            key={item.name}
            draggable
            onDragStart={(e) => onDragStart(e, item)}
            className={cn(
              "flex items-center gap-2.5 px-3 py-2 rounded-lg border cursor-grab",
              "bg-zinc-900/50 transition-colors text-zinc-300 hover:text-zinc-100",
              catColor[item.category],
            )}
          >
            <Icon className="w-4 h-4 shrink-0 text-zinc-500" />
            <span className="text-sm font-medium">{item.name}</span>
            <span className="ml-auto text-[9px] font-mono text-zinc-600">
              {(() => {
                const ins = Object.values(item.inputs);
                const outs = Object.values(item.outputs);
                if (ins[0] && outs[0]) return `${ins[0]} > ${outs[0]}`;
                if (outs[0]) return outs[0];
                if (ins[0]) return ins[0];
                return "";
              })()}
            </span>
          </div>
        );
      })}
    </div>
  );
}

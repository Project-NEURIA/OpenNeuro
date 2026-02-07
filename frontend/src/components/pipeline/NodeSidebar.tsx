import { Mic, AudioLines, MessageSquareText, Brain, Volume2, Radio, Speaker } from "lucide-react";
import { cn } from "@/lib/utils";

const palette = [
  { name: "Microphone", icon: Mic, category: "source", output: "bytes" },
  { name: "VAD", icon: AudioLines, category: "conduit", input: "bytes", output: "bytes" },
  { name: "ASR", icon: MessageSquareText, category: "conduit", input: "bytes", output: "str" },
  { name: "LLM", icon: Brain, category: "conduit", input: "str", output: "str" },
  { name: "TTS", icon: Volume2, category: "conduit", input: "str", output: "bytes" },
  { name: "STS", icon: Radio, category: "conduit", input: "bytes", output: "bytes" },
  { name: "Speaker", icon: Speaker, category: "sink", input: "bytes" },
] as const;

const catColor: Record<string, string> = {
  source: "border-emerald-500/30 hover:border-emerald-500/60",
  conduit: "border-blue-500/30 hover:border-blue-500/60",
  sink: "border-amber-500/30 hover:border-amber-500/60",
};

export function NodeSidebar() {
  function onDragStart(e: React.DragEvent, item: (typeof palette)[number]) {
    e.dataTransfer.setData("application/pipeline-node", JSON.stringify(item));
    e.dataTransfer.effectAllowed = "move";
  }

  return (
    <div className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-950 p-4 flex flex-col gap-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
        Components
      </h2>
      {palette.map((item) => {
        const Icon = item.icon;
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
              {"input" in item ? item.input : ""}
              {"input" in item && "output" in item ? " > " : ""}
              {"output" in item ? item.output : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}

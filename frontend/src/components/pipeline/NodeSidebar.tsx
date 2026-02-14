import { Mic, AudioLines, MessageSquareText, Brain, Volume2, Radio, Speaker, Video, Monitor, Play, Puzzle } from "lucide-react";
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
  VRChatVideo: Video,
  VideoPlayer: Play,
  VideoStream: Monitor,
};

const catAccent: Record<string, { icon: string }> = {
  source: { icon: "text-source/70" },
  conduit: { icon: "text-conduit/70" },
  sink: { icon: "text-sink/70" },
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
    <div
      className={cn(
        "absolute top-4 left-4 z-10 w-52",
        "rounded-2xl border border-glass-border",
        "bg-glass backdrop-blur-xs backdrop-saturate-150",
        "shadow-2xl shadow-black/40",
        "p-3 flex flex-col gap-1.5",
      )}
    >
      <h2 className="text-sm font-semibold text-white px-1 mb-1">
        Components
      </h2>
      {components.map((item) => {
        const Icon = iconMap[item.name] ?? Puzzle;
        const accent = catAccent[item.category] ?? catAccent.conduit!;
        return (
          <div
            key={item.name}
            draggable
            onDragStart={(e) => onDragStart(e, item)}
            className={cn(
              "flex items-center gap-2.5 px-3 py-2.5 rounded-xl cursor-grab",
              "transition-all duration-200",
              "bg-accent hover:bg-glass-hover",
            )}
          >
            <Icon className={cn("w-4 h-4 shrink-0", accent.icon)} />
            <span className="text-[13px] font-medium text-white/80 tracking-tight">
              {item.name}
            </span>
          </div>
        );
      })}
    </div>
  );
}

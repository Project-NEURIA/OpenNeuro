import { Mic, AudioLines, MessageSquareText, Brain, Volume2, Radio, Speaker, Video, Monitor, Play, Camera, Puzzle, FolderOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ComponentInfo } from "@/lib/types";
import type { ProjectSummary } from "@/lib/api";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  Mic,
  VAD: AudioLines,
  ASR: MessageSquareText,
  LLM: Brain,
  TTS: Volume2,
  STS: Radio,
  Speaker,
  Camera,
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
  projects: ProjectSummary[];
  currentProject: string;
}

export function NodeSidebar({ components, projects, currentProject }: NodeSidebarProps) {
  function onDragStart(e: React.DragEvent, item: ComponentInfo) {
    e.dataTransfer.setData("application/graph-node", JSON.stringify(item));
    e.dataTransfer.effectAllowed = "move";
  }

  function onProjectDragStart(e: React.DragEvent, project: ProjectSummary) {
    e.dataTransfer.setData("application/project-node", project.name);
    e.dataTransfer.effectAllowed = "move";
  }

  // Don't show the currently open project as a draggable component
  const otherProjects = projects.filter((p) => p.name !== currentProject);

  return (
    <div
      className={cn(
        "absolute top-4 left-4 z-10 w-52",
        "rounded-2xl border border-glass-border",
        "bg-glass backdrop-blur-xs backdrop-saturate-150",
        "shadow-2xl shadow-black/40",
        "p-3 flex flex-col gap-1.5 max-h-[70vh]",
      )}
    >
      <h2 className="text-sm font-semibold text-white px-1 mb-1 shrink-0">
        Components
      </h2>
      <div className="flex flex-col gap-1.5 overflow-y-auto min-h-0">
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
      {otherProjects.length > 0 && (
        <>
          <div className="border-t border-white/10 my-1" />
          <h2 className="text-sm font-semibold text-white px-1 mb-0.5 shrink-0">
            Projects
          </h2>
          {otherProjects.map((project) => (
            <div
              key={project.name}
              draggable
              onDragStart={(e) => onProjectDragStart(e, project)}
              className={cn(
                "flex items-center gap-2.5 px-3 py-2.5 rounded-xl cursor-grab",
                "transition-all duration-200",
                "bg-accent hover:bg-glass-hover",
              )}
            >
              <FolderOpen className="w-4 h-4 shrink-0 text-purple-400/70" />
              <span className="text-[13px] font-medium text-white/80 tracking-tight">
                {project.name}
              </span>
            </div>
          ))}
        </>
      )}
      </div>
    </div>
  );
}

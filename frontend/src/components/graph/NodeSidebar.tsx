import { useState } from "react";
import { Mic, AudioLines, MessageSquareText, Brain, Volume2, Radio, Speaker, Video, Monitor, Play, Camera, Puzzle, FolderOpen, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ComponentInfo, IOTag, FunctionalityTag } from "@/lib/types";
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

const ioLabels: Record<IOTag, string> = {
  source: "Sources",
  conduit: "Conduits",
  sink: "Sinks",
};

const ioOrder: IOTag[] = ["source", "conduit", "sink"];

const funcLabels: Record<FunctionalityTag, string> = {
  audio: "Audio",
  video: "Video",
  llm: "LLM",
  image: "Image",
  movement: "Movement",
  misc: "Misc",
  other: "Other",
};

const ioAccent: Record<IOTag, string> = {
  source: "text-source",
  conduit: "text-conduit",
  sink: "text-sink",
};

interface NodeSidebarProps {
  components: ComponentInfo[];
  projects: ProjectSummary[];
  currentProject: string;
}

/** Group components by IO → Functionality */
function groupComponents(components: ComponentInfo[]) {
  const groups: Record<IOTag, Record<FunctionalityTag, ComponentInfo[]>> = {
    source: {} as Record<FunctionalityTag, ComponentInfo[]>,
    conduit: {} as Record<FunctionalityTag, ComponentInfo[]>,
    sink: {} as Record<FunctionalityTag, ComponentInfo[]>,
  };

  for (const comp of components) {
    for (const io of comp.tags.io) {
      for (const func of comp.tags.functionality) {
        if (!groups[io][func]) groups[io][func] = [];
        groups[io][func].push(comp);
      }
    }
  }

  return groups;
}

export function NodeSidebar({ components, projects, currentProject }: NodeSidebarProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  function toggle(key: string) {
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function onDragStart(e: React.DragEvent, item: ComponentInfo) {
    e.dataTransfer.setData("text/plain", JSON.stringify({ kind: "component", ...item }));
    e.dataTransfer.effectAllowed = "move";
  }

  function onProjectDragStart(e: React.DragEvent, project: ProjectSummary) {
    e.dataTransfer.setData("text/plain", JSON.stringify({ kind: "project", name: project.name }));
    e.dataTransfer.effectAllowed = "move";
  }

  const otherProjects = projects.filter((p) => p.name !== currentProject);
  const groups = groupComponents(components);

  return (
    <div
      className={cn(
        "absolute top-4 left-4 z-10 w-52",
        "rounded-2xl border border-glass-border",
        "bg-glass backdrop-blur-xs backdrop-saturate-150",
        "shadow-2xl shadow-black/40",
        "p-3 flex flex-col gap-0.5 max-h-[70vh] overflow-hidden",
      )}
    >
      <h2 className="text-sm font-semibold text-white px-1 mb-1 shrink-0">
        Components
      </h2>
      <div className="flex flex-col gap-0.5 overflow-y-auto overscroll-none min-h-0">
        {ioOrder.map((io) => {
          const funcGroups = groups[io];
          const funcKeys = Object.keys(funcGroups) as FunctionalityTag[];
          if (funcKeys.length === 0) return null;

          const ioKey = `io:${io}`;
          const ioCollapsed = collapsed[ioKey];

          return (
            <div key={io}>
              {/* IO section header */}
              <button
                onClick={() => toggle(ioKey)}
                className={cn(
                  "flex items-center gap-1.5 w-full px-1 py-1.5 text-[11px] font-bold uppercase tracking-wider",
                  ioAccent[io],
                )}
              >
                {ioCollapsed ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
                {ioLabels[io]}
              </button>

              {!ioCollapsed && funcKeys.map((func) => {
                const items = funcGroups[func]!;
                const funcKey = `${io}:${func}`;
                const funcCollapsed = collapsed[funcKey];

                return (
                  <div key={func} className="ml-2">
                    {/* Functionality sub-header */}
                    <button
                      onClick={() => toggle(funcKey)}
                      className="flex items-center gap-1.5 w-full px-1 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
                    >
                      {funcCollapsed ? <ChevronRight size={10} /> : <ChevronDown size={10} />}
                      {funcLabels[func]}
                    </button>

                    {!funcCollapsed && items.map((item) => {
                      const Icon = iconMap[item.name] ?? Puzzle;
                      return (
                        <div
                          key={item.name}
                          draggable
                          onDragStart={(e) => onDragStart(e, item)}
                          className={cn(
                            "flex items-center gap-2.5 px-3 py-2 ml-2 cursor-grab",
                            "transition-all duration-200",
                            "hover:bg-glass-hover",
                          )}
                        >
                          <Icon className={cn("w-3.5 h-3.5 shrink-0", ioAccent[io] + "/70")} />
                          <span className="text-[12px] font-medium text-white/80 tracking-tight truncate">
                            {item.name}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                );
              })}
            </div>
          );
        })}

        {/* Projects section */}
        {otherProjects.length > 0 && (
          <div>
            <div className="border-t border-white/10 my-1.5" />
            <button
              onClick={() => toggle("projects")}
              className="flex items-center gap-1.5 w-full px-1 py-1.5 text-[11px] font-bold uppercase tracking-wider text-purple-400"
            >
              {collapsed["projects"] ? <ChevronRight size={12} /> : <ChevronDown size={12} />}
              Projects
            </button>
            {!collapsed["projects"] && otherProjects.map((project) => (
              <div
                key={project.name}
                draggable
                onDragStart={(e) => onProjectDragStart(e, project)}
                className={cn(
                  "flex items-center gap-2.5 px-3 py-2 ml-4 cursor-grab",
                  "transition-all duration-200",
                  "hover:bg-glass-hover",
                )}
              >
                <FolderOpen className="w-3.5 h-3.5 shrink-0 text-purple-400/70" />
                <span className="text-[12px] font-medium text-white/80 tracking-tight">
                  {project.name}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

import { useState, useRef } from "react";
import { Mic, AudioLines, MessageSquareText, Brain, Volume2, Radio, Speaker, Video, Monitor, Play, Camera, Puzzle, FolderOpen, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ComponentInfo, IOTag, FunctionalityTag, GPUTag } from "@/lib/types";
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

const ioTagColors: Record<IOTag, { bg: string; text: string; border: string }> = {
  source: { bg: "bg-source/15", text: "text-source", border: "border-source/30" },
  conduit: { bg: "bg-conduit/15", text: "text-conduit", border: "border-conduit/30" },
  sink: { bg: "bg-sink/15", text: "text-sink", border: "border-sink/30" },
};

const funcTagColors: Record<FunctionalityTag, { bg: string; text: string; border: string }> = {
  audio: { bg: "bg-tag-audio/15", text: "text-tag-audio", border: "border-tag-audio/30" },
  video: { bg: "bg-tag-video/15", text: "text-tag-video", border: "border-tag-video/30" },
  llm: { bg: "bg-tag-llm/15", text: "text-tag-llm", border: "border-tag-llm/30" },
  image: { bg: "bg-tag-image/15", text: "text-tag-image", border: "border-tag-image/30" },
  movement: { bg: "bg-tag-movement/15", text: "text-tag-movement", border: "border-tag-movement/30" },
  misc: { bg: "bg-tag-misc/15", text: "text-tag-misc", border: "border-tag-misc/30" },
  other: { bg: "bg-tag-other/15", text: "text-tag-other", border: "border-tag-other/30" },
};

const gpuTagColors: Record<string, { bg: string; text: string; border: string }> = {
  nvidia: { bg: "bg-tag-nvidia/15", text: "text-tag-nvidia", border: "border-tag-nvidia/30" },
  apple: { bg: "bg-tag-apple/15", text: "text-tag-apple", border: "border-tag-apple/30" },
  intel: { bg: "bg-tag-intel/15", text: "text-tag-intel", border: "border-tag-intel/30" },
  amd: { bg: "bg-tag-amd/15", text: "text-tag-amd", border: "border-tag-amd/30" },
};

/** Render simple inline markdown: **bold**, *italic*, `code` */
function InlineMarkdown({ text }: { text: string }) {
  const parts: React.ReactNode[] = [];
  // Match **bold**, *italic*, `code`
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(<strong key={key++} className="font-bold text-white/90">{match[2]}</strong>);
    } else if (match[3]) {
      parts.push(<em key={key++} className="italic text-white/70">{match[3]}</em>);
    } else if (match[4]) {
      parts.push(<code key={key++} className="px-1 py-0.5 rounded bg-white/[0.08] text-[10px] font-mono text-white/80">{match[4]}</code>);
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return <>{parts}</>;
}

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

function InfoPanel({ item, sidebarRef, y }: { item: ComponentInfo; sidebarRef: React.RefObject<HTMLDivElement | null>; y: number }) {
  const inputs = Object.entries(item.inputs);
  const outputs = Object.entries(item.outputs);
  const rect = sidebarRef.current?.getBoundingClientRect();
  if (!rect) return null;

  return (
    <div
      className={cn(
        "fixed z-50 w-64",
        "bg-glass backdrop-blur-xs backdrop-saturate-150",
        "border border-glass-border rounded-lg",
        "shadow-2xl shadow-black/40",
        "p-3 text-[11px]",
      )}
      style={{ left: rect.right + 8, top: Math.max(8, rect.top + y - 40) }}
    >
      {/* Type */}
      <div className="font-bold text-[13px] text-white mb-1">{item.type_}</div>

      {/* Description */}
      {item.description && (
        <div className="text-muted-foreground mb-2 leading-relaxed text-[11px]">
          <InlineMarkdown text={item.description} />
        </div>
      )}

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mb-2">
        {item.tags.io.map((t) => {
          const colors = ioTagColors[t];
          return (
            <span key={t} className={cn("px-1.5 py-0.5 rounded border uppercase tracking-wider text-[9px] font-semibold", colors.bg, colors.text, colors.border)}>{t}</span>
          );
        })}
        {item.tags.functionality.map((t) => {
          const colors = funcTagColors[t];
          return (
            <span key={t} className={cn("px-1.5 py-0.5 rounded border uppercase tracking-wider text-[9px] font-semibold", colors.bg, colors.text, colors.border)}>{t}</span>
          );
        })}
        {item.tags.gpu.filter((g: GPUTag) => g !== "cpu").map((t: GPUTag) => {
          const colors = gpuTagColors[t] ?? { bg: "bg-white/[0.06]", text: "text-white/60", border: "border-white/10" };
          return (
            <span key={t} className={cn("px-1.5 py-0.5 rounded border uppercase tracking-wider text-[9px] font-semibold", colors.bg, colors.text, colors.border)}>{t}</span>
          );
        })}
      </div>

      {/* IO */}
      {inputs.length > 0 && (
        <div className="mb-1.5">
          <div className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground mb-0.5">Inputs</div>
          {inputs.map(([k, v]) => (
            <div key={k} className="flex gap-1.5 text-white/70">
              <span className="text-muted-foreground">{k}:</span>
              <span>{v}</span>
            </div>
          ))}
        </div>
      )}
      {outputs.length > 0 && (
        <div>
          <div className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground mb-0.5">Outputs</div>
          {outputs.map(([k, v]) => (
            <div key={k} className="flex gap-1.5 text-white/70">
              <span className="text-muted-foreground">{k}:</span>
              <span>{v}</span>
            </div>
          ))}
        </div>
      )}

      {/* Init params */}
      {Object.keys(item.init).length > 0 && (
        <div className="mt-1.5">
          <div className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground mb-0.5">Config</div>
          {Object.keys(item.init).map((k) => (
            <div key={k} className="text-white/70">{k}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export function NodeSidebar({ components, projects, currentProject }: NodeSidebarProps) {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [hovered, setHovered] = useState<{ item: ComponentInfo; y: number } | null>(null);
  const hoverTimeout = useRef<ReturnType<typeof setTimeout>>();
  const sidebarRef = useRef<HTMLDivElement>(null);

  function toggle(key: string) {
    setCollapsed((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function onDragStart(e: React.DragEvent, item: ComponentInfo) {
    const data = JSON.stringify({ kind: "component", ...item });
    e.dataTransfer.setData("text/plain", data);
    e.dataTransfer.setData("application/openneuro", data);
    e.dataTransfer.effectAllowed = "move";
    setHovered(null);
  }

  function onProjectDragStart(e: React.DragEvent, project: ProjectSummary) {
    const data = JSON.stringify({ kind: "project", name: project.name });
    e.dataTransfer.setData("text/plain", data);
    e.dataTransfer.setData("application/openneuro", data);
    e.dataTransfer.effectAllowed = "move";
  }

  function onItemEnter(e: React.MouseEvent, item: ComponentInfo) {
    clearTimeout(hoverTimeout.current);
    const sidebarRect = sidebarRef.current?.getBoundingClientRect();
    const y = e.currentTarget.getBoundingClientRect().top - (sidebarRect?.top ?? 0);
    hoverTimeout.current = setTimeout(() => setHovered({ item, y }), 300);
  }

  function onItemLeave() {
    clearTimeout(hoverTimeout.current);
    hoverTimeout.current = setTimeout(() => setHovered(null), 100);
  }

  const otherProjects = projects.filter((p) => p.name !== currentProject);
  const groups = groupComponents(components);

  return (
    <>
    <div
      ref={sidebarRef}
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
                    <button
                      onClick={() => toggle(funcKey)}
                      className="flex items-center gap-1.5 w-full px-1 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
                    >
                      {funcCollapsed ? <ChevronRight size={10} /> : <ChevronDown size={10} />}
                      {funcLabels[func]}
                    </button>

                    {!funcCollapsed && items.map((item) => {
                      const Icon = iconMap[item.type_] ?? Puzzle;
                      return (
                        <div
                          key={item.type_}
                          draggable
                          onDragStart={(e) => onDragStart(e, item)}
                          onMouseEnter={(e) => onItemEnter(e, item)}
                          onMouseLeave={onItemLeave}
                          className={cn(
                            "flex items-center gap-2.5 px-3 py-2 ml-2 cursor-grab",
                            "transition-all duration-200",
                            "hover:bg-glass-hover",
                          )}
                        >
                          <Icon className={cn("w-3.5 h-3.5 shrink-0", ioAccent[io] + "/70")} />
                          <span className="text-[12px] font-medium text-white/80 tracking-tight truncate">
                            {item.type_}
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

    {/* Hover info panel — rendered outside sidebar to avoid overflow clip */}
    {hovered && (
      <div
        onMouseEnter={() => clearTimeout(hoverTimeout.current)}
        onMouseLeave={onItemLeave}
      >
        <InfoPanel
          item={hovered.item}
          sidebarRef={sidebarRef}
          y={hovered.y}
        />
      </div>
    )}
  </>
  );
}

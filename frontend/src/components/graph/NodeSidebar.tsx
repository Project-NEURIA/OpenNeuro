import { useState, useRef } from "react";
import { Mic, AudioLines, MessageSquareText, Brain, Volume2, Radio, Speaker, Video, Monitor, Play, Camera, Puzzle, FolderOpen, ChevronDown, ChevronRight, Search, X } from "lucide-react";
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
        <div className="text-muted-foreground mb-2 leading-relaxed">{item.description}</div>
      )}

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mb-2">
        {item.tags.io.map((t) => (
          <span key={t} className="px-1.5 py-0.5 bg-white/[0.06] text-white/60 uppercase tracking-wider text-[9px] font-semibold">{t}</span>
        ))}
        {item.tags.functionality.map((t) => (
          <span key={t} className="px-1.5 py-0.5 bg-white/[0.06] text-white/60 uppercase tracking-wider text-[9px] font-semibold">{t}</span>
        ))}
        {item.tags.gpu.filter((g) => g !== "cpu").map((t) => (
          <span key={t} className="px-1.5 py-0.5 bg-white/[0.06] text-white/60 uppercase tracking-wider text-[9px] font-semibold">{t}</span>
        ))}
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
  const [search, setSearch] = useState("");
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
  const query = search.toLowerCase();
  const filtered = query
    ? components.filter((c) => c.type_.toLowerCase().includes(query))
    : components;
  const groups = groupComponents(filtered);

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
      <div className="relative shrink-0 mx-0.5 mb-1">
        <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground pointer-events-none" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search..."
          className={cn(
            "w-full pl-7 pr-7 py-1.5 text-[11px] text-white/90 placeholder:text-muted-foreground",
            "bg-white/[0.06] border border-glass-border rounded-lg",
            "outline-none focus:border-white/20 transition-colors",
          )}
        />
        {search && (
          <button
            onClick={() => setSearch("")}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-white transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>
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

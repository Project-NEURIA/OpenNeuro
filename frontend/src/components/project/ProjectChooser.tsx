import { useCallback, useEffect, useRef, useState } from "react";
import { Plus, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  fetchProjects,
  createProject,
  deleteProject,
  startProject,
  projectThumbnailUrl,
  type ProjectSummary,
} from "@/lib/api";

interface ProjectChooserProps {
  hadProject: boolean;
  onOpen: (name: string) => void;
  onCancel: () => void;
}

export function ProjectChooser({
  hadProject,
  onOpen,
  onCancel,
}: ProjectChooserProps) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    fetchProjects().then(setProjects).catch(console.error);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (showCreateModal) inputRef.current?.focus();
  }, [showCreateModal]);

  const handleOpen = useCallback(
    async (name: string) => {
      await startProject(name);
      onOpen(name);
    },
    [onOpen],
  );

  const handleCreate = useCallback(async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      await createProject(name);
      await startProject(name);
      onOpen(name);
    } catch (err) {
      console.error(err);
      setCreating(false);
    }
  }, [newName, onOpen]);

  const handleDelete = useCallback(
    async (name: string, e: React.MouseEvent) => {
      e.stopPropagation();
      await deleteProject(name);
      refresh();
    },
    [refresh],
  );

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background text-foreground">
      <div data-tauri-drag-region className="absolute top-0 left-0 right-0 h-[15px]" />
      {/* Header */}
      <div className="flex items-center justify-center pt-16 pb-10">
        {hadProject && (
          <button
            onClick={onCancel}
            className={cn(
              "absolute top-5 right-5 p-1.5 rounded-lg",
              "text-muted-foreground transition-colors",
              "hover:text-foreground hover:bg-glass-hover",
            )}
          >
            <X className="w-5 h-5" />
          </button>
        )}
        <h1
          className="text-2xl font-bold tracking-tight"
          style={{ color: "color(display-p3 1.4 1.4 1.4)" }}
        >
          Choose a Project
        </h1>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto px-12">
        <div className="mx-auto grid max-w-5xl grid-cols-4 gap-4">
          {projects.map((p) => (
            <div
              key={p.name}
              className={cn(
                "group relative flex flex-col items-center rounded-2xl border p-4 transition-all duration-200",
                "bg-glass backdrop-blur-xs backdrop-saturate-150",
                "border-glass-border hover:border-conduit/30 hover:bg-glass-hover",
              )}
            >
              <button
                type="button"
                className="flex w-full flex-col items-center"
                onClick={() => handleOpen(p.name)}
              >
                <div className="mb-3 h-28 w-full overflow-hidden rounded-xl bg-accent">
                  {p.has_thumbnail && (
                    <img
                      src={projectThumbnailUrl(p.name)}
                      alt={p.name}
                      className="h-full w-full object-cover"
                    />
                  )}
                </div>
                <span className="text-[13px] font-medium truncate w-full text-center tracking-tight text-white/80">
                  {p.name}
                </span>
              </button>
              <button
                type="button"
                className={cn(
                  "absolute top-2.5 right-2.5 p-1.5 rounded-lg",
                  "opacity-0 transition-all duration-200 group-hover:opacity-100",
                  "text-muted-foreground hover:text-destructive-foreground hover:bg-destructive/30",
                )}
                onClick={(e) => handleDelete(p.name, e)}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}

          {/* New project card */}
          <button
            className={cn(
              "flex flex-col items-center justify-center rounded-2xl border border-dashed p-4 transition-all duration-200",
              "border-muted-foreground/20 hover:border-conduit/30 hover:bg-glass-hover",
            )}
            onClick={() => {
              setNewName("");
              setShowCreateModal(true);
            }}
          >
            <div className="flex h-28 w-full items-center justify-center rounded-xl">
              <Plus className="w-8 h-8 text-muted-foreground/50" />
            </div>
            <span className="text-[13px] text-muted-foreground/60 tracking-tight">
              New project
            </span>
          </button>
        </div>
      </div>

      {/* Create project modal */}
      {showCreateModal && (
        <div
          className="fixed inset-0 z-60 flex items-center justify-center bg-black/60 backdrop-blur-xs"
          onClick={() => setShowCreateModal(false)}
        >
          <div
            className={cn(
              "w-80 rounded-2xl border border-glass-border p-5",
              "bg-glass backdrop-blur-xs backdrop-saturate-150",
              "shadow-2xl shadow-black/40",
              "flex flex-col gap-4",
            )}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-sm font-semibold text-white">New Project</h2>
            <input
              ref={inputRef}
              type="text"
              placeholder="Project name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreate();
                if (e.key === "Escape") setShowCreateModal(false);
              }}
              className={cn(
                "h-9 w-full rounded-xl px-3 text-[13px]",
                "bg-accent border border-glass-border",
                "text-foreground placeholder:text-muted-foreground/50",
                "focus:border-conduit/40 focus:outline-none transition-colors",
              )}
            />
            <div className="flex items-center justify-end gap-2">
              <button
                onClick={() => setShowCreateModal(false)}
                className={cn(
                  "h-8 rounded-xl px-4 text-[13px]",
                  "text-muted-foreground transition-colors",
                  "hover:bg-glass-hover hover:text-foreground",
                )}
              >
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={!newName.trim() || creating}
                className={cn(
                  "h-8 rounded-xl px-4 text-[13px] font-medium transition-all duration-200",
                  "bg-conduit/20 text-conduit",
                  "hover:bg-conduit/30",
                  "disabled:opacity-30 disabled:cursor-not-allowed",
                )}
              >
                {creating ? "Creating..." : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

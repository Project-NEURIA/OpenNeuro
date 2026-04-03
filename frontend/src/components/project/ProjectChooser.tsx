import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Plus, Share2, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  fetchProjects,
  createProject,
  deleteProject,
  startProject,
  exportProject,
  importProject,
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
  const [showImportModal, setShowImportModal] = useState(false);
  const [showExportResult, setShowExportResult] = useState<{
    dir: string;
    assets: string[];
  } | null>(null);
  const [newName, setNewName] = useState("");
  const [importUrl, setImportUrl] = useState("");
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const importInputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    fetchProjects().then(setProjects).catch(console.error);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (showCreateModal) inputRef.current?.focus();
  }, [showCreateModal]);

  useEffect(() => {
    if (showImportModal) importInputRef.current?.focus();
  }, [showImportModal]);

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

  const handleExport = useCallback(
    async (name: string, e: React.MouseEvent) => {
      e.stopPropagation();
      try {
        const result = await exportProject(name);
        setShowExportResult({ dir: result.project_dir, assets: result.assets_copied });
      } catch (err) {
        console.error(err);
      }
    },
    [],
  );

  const handleImport = useCallback(async () => {
    const url = importUrl.trim();
    if (!url) return;
    setImporting(true);
    try {
      const result = await importProject(url);
      refresh();
      setShowImportModal(false);
      setImportUrl("");
      await startProject(result.name);
      onOpen(result.name);
    } catch (err) {
      console.error(err);
    } finally {
      setImporting(false);
    }
  }, [importUrl, refresh, onOpen]);

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
            <button
              key={p.name}
              className={cn(
                "group relative flex flex-col items-center rounded-2xl border p-4 transition-all duration-200",
                "bg-glass backdrop-blur-xs backdrop-saturate-150",
                "border-glass-border hover:border-conduit/30 hover:bg-glass-hover",
              )}
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
              <div className="absolute top-2.5 right-2.5 flex gap-1 opacity-0 transition-all duration-200 group-hover:opacity-100">
                <button
                  className={cn(
                    "p-1.5 rounded-lg",
                    "text-muted-foreground hover:text-conduit hover:bg-conduit/20",
                  )}
                  onClick={(e) => handleExport(p.name, e)}
                  title="Share project"
                >
                  <Share2 className="w-3.5 h-3.5" />
                </button>
                <button
                  className={cn(
                    "p-1.5 rounded-lg",
                    "text-muted-foreground hover:text-destructive-foreground hover:bg-destructive/30",
                  )}
                  onClick={(e) => handleDelete(p.name, e)}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </button>
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

          {/* Import project card */}
          <button
            className={cn(
              "flex flex-col items-center justify-center rounded-2xl border border-dashed p-4 transition-all duration-200",
              "border-muted-foreground/20 hover:border-conduit/30 hover:bg-glass-hover",
            )}
            onClick={() => {
              setImportUrl("");
              setShowImportModal(true);
            }}
          >
            <div className="flex h-28 w-full items-center justify-center rounded-xl">
              <Download className="w-8 h-8 text-muted-foreground/50" />
            </div>
            <span className="text-[13px] text-muted-foreground/60 tracking-tight">
              Import project
            </span>
          </button>
        </div>
      </div>

      {/* Import project modal */}
      {showImportModal && (
        <div
          className="fixed inset-0 z-60 flex items-center justify-center bg-black/60 backdrop-blur-xs"
          onClick={() => setShowImportModal(false)}
        >
          <div
            className={cn(
              "w-96 rounded-2xl border border-glass-border p-5",
              "bg-glass backdrop-blur-xs backdrop-saturate-150",
              "shadow-2xl shadow-black/40",
              "flex flex-col gap-4",
            )}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-sm font-semibold text-white">Import Project</h2>
            <p className="text-[12px] text-muted-foreground">
              Enter a git URL to clone a shared project.
            </p>
            <input
              ref={importInputRef}
              type="text"
              placeholder="https://github.com/user/project.git"
              value={importUrl}
              onChange={(e) => setImportUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleImport();
                if (e.key === "Escape") setShowImportModal(false);
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
                onClick={() => setShowImportModal(false)}
                className={cn(
                  "h-8 rounded-xl px-4 text-[13px]",
                  "text-muted-foreground transition-colors",
                  "hover:bg-glass-hover hover:text-foreground",
                )}
              >
                Cancel
              </button>
              <button
                onClick={handleImport}
                disabled={!importUrl.trim() || importing}
                className={cn(
                  "h-8 rounded-xl px-4 text-[13px] font-medium transition-all duration-200",
                  "bg-conduit/20 text-conduit",
                  "hover:bg-conduit/30",
                  "disabled:opacity-30 disabled:cursor-not-allowed",
                )}
              >
                {importing ? "Importing..." : "Import"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export result modal */}
      {showExportResult && (
        <div
          className="fixed inset-0 z-60 flex items-center justify-center bg-black/60 backdrop-blur-xs"
          onClick={() => setShowExportResult(null)}
        >
          <div
            className={cn(
              "w-96 rounded-2xl border border-glass-border p-5",
              "bg-glass backdrop-blur-xs backdrop-saturate-150",
              "shadow-2xl shadow-black/40",
              "flex flex-col gap-4",
            )}
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-sm font-semibold text-white">Project Exported</h2>
            <p className="text-[12px] text-muted-foreground">
              Assets have been copied and paths rewritten. Your project is ready to share.
            </p>
            <div className="rounded-xl bg-accent border border-glass-border p-3">
              <p className="text-[12px] text-muted-foreground mb-1">Project directory:</p>
              <p className="text-[12px] text-white font-mono break-all">
                {showExportResult.dir}
              </p>
            </div>
            {showExportResult.assets.length > 0 && (
              <div className="rounded-xl bg-accent border border-glass-border p-3">
                <p className="text-[12px] text-muted-foreground mb-1">
                  Assets copied ({showExportResult.assets.length}):
                </p>
                <ul className="text-[11px] text-white/70 font-mono space-y-0.5">
                  {showExportResult.assets.map((a) => (
                    <li key={a}>{a}</li>
                  ))}
                </ul>
              </div>
            )}
            <div className="flex justify-end">
              <button
                onClick={() => setShowExportResult(null)}
                className={cn(
                  "h-8 rounded-xl px-4 text-[13px] font-medium transition-all duration-200",
                  "bg-conduit/20 text-conduit",
                  "hover:bg-conduit/30",
                )}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

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

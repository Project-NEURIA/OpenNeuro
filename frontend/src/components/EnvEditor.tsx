import { useCallback, useEffect, useState } from "react";
import { X, Plus, Trash2, Eye, EyeOff, Save } from "lucide-react";
import { fetchEnv, putEnv } from "@/lib/api";

interface EnvEntry {
  key: string;
  value: string;
}

function parseEnv(raw: string): EnvEntry[] {
  return raw
    .split("\n")
    .filter((line) => line.trim() && !line.startsWith("#"))
    .map((line) => {
      const eq = line.indexOf("=");
      if (eq === -1) return { key: line.trim(), value: "" };
      return { key: line.slice(0, eq).trim(), value: line.slice(eq + 1).trim() };
    });
}

function serializeEnv(entries: EnvEntry[]): string {
  return entries
    .filter((e) => e.key.trim())
    .map((e) => `${e.key}=${e.value}`)
    .join("\n");
}

interface EnvEditorProps {
  onClose: () => void;
}

export function EnvEditor({ onClose }: EnvEditorProps) {
  const [entries, setEntries] = useState<EnvEntry[]>([]);
  const [revealed, setRevealed] = useState<Set<number>>(new Set());
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetchEnv()
      .then((raw) => {
        const parsed = parseEnv(raw);
        setEntries(parsed.length > 0 ? parsed : [{ key: "", value: "" }]);
      })
      .catch(console.error);
  }, []);

  const handleSave = useCallback(async () => {
    await putEnv(serializeEnv(entries));
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }, [entries]);

  const updateEntry = (index: number, field: "key" | "value", val: string) => {
    setEntries((prev) => prev.map((e, i) => (i === index ? { ...e, [field]: val } : e)));
  };

  const addEntry = () => {
    setEntries((prev) => [...prev, { key: "", value: "" }]);
  };

  const removeEntry = (index: number) => {
    setEntries((prev) => prev.filter((_, i) => i !== index));
    setRevealed((prev) => {
      const next = new Set<number>();
      for (const i of prev) {
        if (i < index) next.add(i);
        else if (i > index) next.add(i - 1);
      }
      return next;
    });
  };

  const toggleReveal = (index: number) => {
    setRevealed((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="flex w-[520px] max-h-[80vh] flex-col rounded-xl border border-[var(--border)] bg-[var(--glass)] shadow-2xl backdrop-blur">
        <div className="flex items-center justify-between border-b border-[var(--border)] px-4 py-3">
          <h2 className="text-sm font-medium text-[var(--foreground)]">
            Environment Variables
          </h2>
          <button
            onClick={onClose}
            className="rounded p-1 text-[var(--muted-foreground)] hover:bg-[var(--glass-hover)] hover:text-[var(--foreground)]"
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
          {entries.map((entry, i) => (
            <div key={i} className="flex items-center gap-2">
              <input
                value={entry.key}
                onChange={(e) => updateEntry(i, "key", e.target.value)}
                placeholder="KEY"
                spellCheck={false}
                className="w-[160px] shrink-0 rounded-md border border-[var(--border)] bg-transparent px-2 py-1.5 font-mono text-xs text-[var(--foreground)] placeholder:text-[var(--muted-foreground)]/40 focus:outline-none focus:border-[var(--accent)]"
              />
              <div className="relative flex-1">
                <input
                  type={revealed.has(i) ? "text" : "password"}
                  value={entry.value}
                  onChange={(e) => updateEntry(i, "value", e.target.value)}
                  placeholder="value"
                  spellCheck={false}
                  className="w-full rounded-md border border-[var(--border)] bg-transparent px-2 py-1.5 pr-8 font-mono text-xs text-[var(--foreground)] placeholder:text-[var(--muted-foreground)]/40 focus:outline-none focus:border-[var(--accent)]"
                />
                <button
                  onClick={() => toggleReveal(i)}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
                >
                  {revealed.has(i) ? <EyeOff size={12} /> : <Eye size={12} />}
                </button>
              </div>
              <button
                onClick={() => removeEntry(i)}
                className="rounded p-1 text-[var(--muted-foreground)] hover:text-red-400"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <div className="flex items-center justify-between border-t border-[var(--border)] px-4 py-3">
          <button
            onClick={addEntry}
            className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs text-[var(--muted-foreground)] transition-colors hover:bg-[var(--glass-hover)] hover:text-[var(--foreground)]"
          >
            <Plus size={14} />
            Add
          </button>
          <div className="flex items-center gap-2">
            {saved && <span className="text-xs text-green-400">Saved</span>}
            <button
              onClick={handleSave}
              className="flex items-center gap-1.5 rounded-lg bg-[var(--accent)] px-3 py-1.5 text-xs font-medium text-[var(--foreground)] transition-colors hover:brightness-110"
            >
              <Save size={14} />
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

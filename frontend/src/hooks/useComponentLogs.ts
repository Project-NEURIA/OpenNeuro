import { useEffect, useRef, useState } from "react";
import { fetchComponentLogs, type ComponentLogEntry } from "@/lib/api";

export function useComponentLogs(nodeId: string | null) {
  const [entries, setEntries] = useState<ComponentLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const lastSeqRef = useRef(0);

  useEffect(() => {
    setEntries([]);
    setError(null);
    lastSeqRef.current = 0;

    if (!nodeId) return;

    let cancelled = false;

    async function pull() {
      try {
        const response = await fetchComponentLogs(nodeId, lastSeqRef.current, 300);
        if (cancelled) return;
        if (response.entries.length === 0) return;

        lastSeqRef.current = response.entries[response.entries.length - 1]!.seq;
        setEntries((prev) => {
          const next = [...prev, ...response.entries];
          return next.slice(-600);
        });
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load logs");
      }
    }

    void pull();
    const timer = window.setInterval(() => {
      void pull();
    }, 500);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [nodeId]);

  return { entries, error };
}

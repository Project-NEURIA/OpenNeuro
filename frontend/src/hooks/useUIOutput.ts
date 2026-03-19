import { useEffect, useState } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

/**
 * Generic hook that subscribes to a UI output channel and returns the latest payload.
 * Works with any payload type — text strings, JSON objects (BaseModel), or ArrayBuffer (binary).
 */
export function useUIOutput<T = unknown>(
  nodeId: string | null,
  channel: string,
): T | null {
  const { subscribe } = useUIChannel();
  const [value, setValue] = useState<T | null>(null);

  useEffect(() => {
    if (!nodeId) return;

    const unsub = subscribe(nodeId, channel, (payload: unknown) => {
      setValue(payload as T);
    });

    return () => {
      unsub();
      setValue(null);
    };
  }, [nodeId, channel, subscribe]);

  return value;
}

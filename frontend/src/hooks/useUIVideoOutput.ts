import { useEffect, useRef, useState } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

/**
 * Subscribes to a UIVideoSender channel and returns the latest frame as an object URL.
 */
export function useUIVideoOutput(
  nodeId: string | null,
  channel: string,
): string | null {
  const { subscribe } = useUIChannel();
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const prevUrl = useRef<string | null>(null);

  useEffect(() => {
    if (!nodeId) return;

    const unsub = subscribe(nodeId, channel, (payload: unknown) => {
      if (payload instanceof ArrayBuffer) {
        const blob = new Blob([payload], { type: "image/jpeg" });
        if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
        const url = URL.createObjectURL(blob);
        prevUrl.current = url;
        setFrameUrl(url);
      }
    });

    return () => {
      unsub();
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
      prevUrl.current = null;
      setFrameUrl(null);
    };
  }, [nodeId, channel, subscribe]);

  return frameUrl;
}

import { useEffect, useRef, useState } from "react";

/**
 * Polls a VideoStream node's frame endpoint and returns the latest frame as an object URL.
 */
export function useVideoStream(nodeId: string | null): string | null {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const prevUrl = useRef<string | null>(null);

  useEffect(() => {
    if (!nodeId) return;

    let active = true;

    async function poll() {
      while (active) {
        try {
          const res = await fetch(`/video/${nodeId}/frame`);
          if (res.ok && active) {
            const blob = await res.blob();
            if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
            const url = URL.createObjectURL(blob);
            prevUrl.current = url;
            setFrameUrl(url);
          }
        } catch {
          // ignore fetch errors
        }
        await new Promise((r) => setTimeout(r, 33));
      }
    }

    poll();

    return () => {
      active = false;
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
      prevUrl.current = null;
      setFrameUrl(null);
    };
  }, [nodeId]);

  return frameUrl;
}

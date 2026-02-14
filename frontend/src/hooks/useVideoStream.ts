import { useEffect, useRef, useState } from "react";

/**
 * Connects to a VideoStream node's WebSocket and returns the latest frame as an object URL.
 * Automatically cleans up the previous object URL on each new frame.
 */
export function useVideoStream(nodeId: string | null): string | null {
  const [frameUrl, setFrameUrl] = useState<string | null>(null);
  const prevUrl = useRef<string | null>(null);

  useEffect(() => {
    if (!nodeId) return;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/video/ws/${nodeId}`);
    ws.binaryType = "arraybuffer";

    ws.onmessage = (e) => {
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
      const blob = new Blob([e.data], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      prevUrl.current = url;
      setFrameUrl(url);
    };

    return () => {
      ws.close();
      if (prevUrl.current) URL.revokeObjectURL(prevUrl.current);
      prevUrl.current = null;
      setFrameUrl(null);
    };
  }, [nodeId]);

  return frameUrl;
}

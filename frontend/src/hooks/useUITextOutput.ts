import { useEffect, useState } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

/**
 * Subscribes to a UITextSender channel and returns the latest text payload.
 */
export function useUITextOutput(
  nodeId: string | null,
  channel: string,
): string | null {
  const { subscribe } = useUIChannel();
  const [text, setText] = useState<string | null>(null);

  useEffect(() => {
    if (!nodeId) return;

    const unsub = subscribe(nodeId, channel, (payload: unknown) => {
      if (typeof payload === "string") {
        setText(payload);
      }
    });

    return () => {
      unsub();
      setText(null);
    };
  }, [nodeId, channel, subscribe]);

  return text;
}

import { useCallback } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

/**
 * Generic hook that returns a send function for a UI input channel.
 * Works with any payload type — strings, objects (BaseModel), etc.
 */
export function useUIInput<T = unknown>(
  nodeId: string | null,
  channel: string,
): (data: T) => void {
  const { sendUIInput } = useUIChannel();

  return useCallback(
    (data: T) => {
      if (nodeId) {
        sendUIInput(nodeId, channel, data);
      }
    },
    [nodeId, channel, sendUIInput],
  );
}

import { useCallback, useEffect, useRef, useState } from "react";

export type UIOutputCallback = (payload: unknown) => void;

export interface UIChannelManager {
  /** Send text input from frontend to a component's UIReceiver */
  sendUIInput: (nodeId: string, channel: string, payload: unknown) => void;
  /** Subscribe to output from a component's UISender */
  subscribe: (
    nodeId: string,
    channel: string,
    callback: UIOutputCallback,
  ) => () => void;
  connected: boolean;
}

/**
 * Manages a single WebSocket connection to /ui/ws for all UI channels.
 */
export function useUIChannelManager(): UIChannelManager {
  const wsRef = useRef<WebSocket | null>(null);
  const listenersRef = useRef<Map<string, Set<UIOutputCallback>>>(new Map());
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const url = `ws://localhost:8000/ui/ws`;

    function connect() {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.binaryType = "arraybuffer";

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        // Reconnect after a short delay
        setTimeout(connect, 1000);
      };

      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          // Text frame: JSON message
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "ui_output") {
              const key = `${msg.node_id}:${msg.channel}`;
              const cbs = listenersRef.current.get(key);
              if (cbs) {
                for (const cb of cbs) cb(msg.payload);
              }
            }
          } catch {
            // ignore parse errors
          }
        } else {
          // Binary frame: 2-byte header length + JSON header + payload
          const buf = event.data as ArrayBuffer;
          const view = new DataView(buf);
          const headerLen = view.getUint16(0);
          const headerBytes = new Uint8Array(buf, 2, headerLen);
          const header = JSON.parse(new TextDecoder().decode(headerBytes));
          const payload = buf.slice(2 + headerLen);

          const key = `${header.node_id}:${header.channel}`;
          const cbs = listenersRef.current.get(key);
          if (cbs) {
            for (const cb of cbs) cb(payload);
          }
        }
      };
    }

    connect();

    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  const sendUIInput = useCallback(
    (nodeId: string, channel: string, payload: unknown) => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "ui_input",
            node_id: nodeId,
            channel,
            payload,
          }),
        );
      }
    },
    [],
  );

  const subscribe = useCallback(
    (
      nodeId: string,
      channel: string,
      callback: UIOutputCallback,
    ): (() => void) => {
      const key = `${nodeId}:${channel}`;
      let set = listenersRef.current.get(key);
      if (!set) {
        set = new Set();
        listenersRef.current.set(key, set);
      }
      set.add(callback);
      return () => {
        set!.delete(callback);
        if (set!.size === 0) {
          listenersRef.current.delete(key);
        }
      };
    },
    [],
  );

  return { sendUIInput, subscribe, connected };
}

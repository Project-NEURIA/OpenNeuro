import { useCallback, useEffect, useRef, useState } from "react";
import { WS_BASE } from "@/lib/api";

export type UIOutputCallback = (payload: unknown) => void;

export interface UIChannelManager {
  /** Send text input from frontend to a component's UIReceiver */
  sendUIInput: (nodeId: string, channel: string, payload: unknown) => void;
  /** Send binary input from frontend to a component's UIReceiver */
  sendUIBinary: (nodeId: string, channel: string, payload: ArrayBuffer | Uint8Array) => void;
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
    let isMounted = true;
    let reconnectTimeout: number | undefined;
    const url = `${WS_BASE}/ui/ws`;

    function connect() {
      if (!isMounted) return;
      
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        if (!isMounted) return;
        setConnected(true);
      };
      
      ws.onclose = () => {
        if (!isMounted) return;
        setConnected(false);
        // Reconnect after a short delay
        reconnectTimeout = window.setTimeout(connect, 1000);
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
      isMounted = false;
      if (reconnectTimeout !== undefined) {
        window.clearTimeout(reconnectTimeout);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
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

  const sendUIBinary = useCallback(
    (nodeId: string, channel: string, payload: ArrayBuffer | Uint8Array) => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        const header = JSON.stringify({ type: "ui_input", node_id: nodeId, channel });
        const headerBytes = new TextEncoder().encode(header);
        
        const buffer = new Uint8Array(2 + headerBytes.length + payload.byteLength);
        
        // 2-byte header length (big-endian)
        const view = new DataView(buffer.buffer);
        view.setUint16(0, headerBytes.length, false);
        
        buffer.set(headerBytes, 2);
        buffer.set(new Uint8Array(payload instanceof ArrayBuffer ? payload : payload.buffer), 2 + headerBytes.length);
        
        ws.send(buffer);
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

  return { sendUIInput, sendUIBinary, subscribe, connected };
}

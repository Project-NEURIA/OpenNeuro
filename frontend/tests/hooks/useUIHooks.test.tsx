import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useUIChannelManager } from "@/hooks/useUIChannel";
import { useUIInput } from "@/hooks/useUIInput";
import { useUIOutput } from "@/hooks/useUIOutput";
import { useUITextOutput } from "@/hooks/useUITextOutput";
import { useUIVideoOutput } from "@/hooks/useUIVideoOutput";
import * as channelContext from "@/contexts/UIChannelContext";

class MockWebSocket {
  static OPEN = 1;
  static instances: MockWebSocket[] = [];
  readyState = 0;
  binaryType = "";
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  sent: string[] = [];
  close = vi.fn();

  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  send(payload: string) {
    this.sent.push(payload);
  }
}

describe("useUIChannelManager", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.restoreAllMocks();
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
  });

  it("connects, dispatches text and binary messages, sends input, unsubscribes, reconnects, and cleans up", async () => {
    const { result, unmount } = renderHook(() => useUIChannelManager());
    const socket = MockWebSocket.instances[0]!;

    expect(socket.url).toBe("ws://localhost:3000/ui/ws");
    expect(socket.binaryType).toBe("arraybuffer");

    socket.readyState = MockWebSocket.OPEN;
    act(() => {
      socket.onopen?.();
    });
    expect(result.current.connected).toBe(true);

    const textHandler = vi.fn();
    const binaryHandler = vi.fn();
    const unsubscribeText = result.current.subscribe("node-1", "text", textHandler);
    const unsubscribeBinary = result.current.subscribe("node-1", "video", binaryHandler);
    const secondTextHandler = vi.fn();
    const unsubscribeSecondText = result.current.subscribe("node-1", "text", secondTextHandler);

    await act(async () => {
      socket.onmessage?.({
        data: JSON.stringify({
          type: "ui_output",
          node_id: "node-1",
          channel: "text",
          payload: { ok: true },
        }),
      } as MessageEvent);
    });
    expect(textHandler).toHaveBeenCalledWith({ ok: true });

    await act(async () => {
      socket.onmessage?.({ data: "{bad json" } as MessageEvent);
    });
    expect(textHandler).toHaveBeenCalledTimes(1);

    await act(async () => {
      socket.onmessage?.({
        data: JSON.stringify({
          type: "ignored",
          node_id: "node-1",
          channel: "text",
          payload: "noop",
        }),
      } as MessageEvent);
    });
    expect(textHandler).toHaveBeenCalledTimes(1);

    const header = new TextEncoder().encode(
      JSON.stringify({ node_id: "node-1", channel: "video" }),
    );
    const payload = new Uint8Array([1, 2, 3, 4]);
    const buffer = new Uint8Array(2 + header.length + payload.length);
    new DataView(buffer.buffer).setUint16(0, header.length);
    buffer.set(header, 2);
    buffer.set(payload, 2 + header.length);

    await act(async () => {
      socket.onmessage?.({ data: buffer.buffer } as MessageEvent);
    });
    expect(binaryHandler).toHaveBeenCalledWith(expect.any(ArrayBuffer));

    unsubscribeBinary();
    await act(async () => {
      socket.onmessage?.({ data: buffer.buffer } as MessageEvent);
    });
    expect(binaryHandler).toHaveBeenCalledTimes(1);

    await act(async () => {
      result.current.sendUIInput("node-1", "text", { ping: true });
    });
    expect(socket.sent).toEqual([
      JSON.stringify({
        type: "ui_input",
        node_id: "node-1",
        channel: "text",
        payload: { ping: true },
      }),
    ]);

    socket.readyState = 0;
    await act(async () => {
      result.current.sendUIInput("node-1", "text", "ignored");
    });
    expect(socket.sent).toHaveLength(1);

    unsubscribeText();
    unsubscribeSecondText();
    unsubscribeBinary();

    await act(async () => {
      socket.onmessage?.({
        data: JSON.stringify({
          type: "ui_output",
          node_id: "node-1",
          channel: "text",
          payload: "again",
        }),
      } as MessageEvent);
    });
    expect(textHandler).toHaveBeenCalledTimes(1);

    await act(async () => {
      socket.onclose?.();
    });
    expect(result.current.connected).toBe(false);

    await act(async () => {
      vi.advanceTimersByTime(1_000);
      await Promise.resolve();
    });
    expect(MockWebSocket.instances).toHaveLength(2);

    unmount();
    expect(MockWebSocket.instances[1]!.close).toHaveBeenCalled();
  });
});

describe("ui input/output hooks", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("sends input only when a node id exists", () => {
    const sendUIInput = vi.fn();
    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput,
      subscribe: vi.fn(() => vi.fn()),
    });

    const withNode = renderHook(() => useUIInput("node-1", "chat"));
    act(() => withNode.result.current("hello"));
    expect(sendUIInput).toHaveBeenCalledWith("node-1", "chat", "hello");

    const withoutNode = renderHook(() => useUIInput<string>(null, "chat"));
    act(() => withoutNode.result.current("ignored"));
    expect(sendUIInput).toHaveBeenCalledTimes(1);
  });

  it("subscribes and unsubscribes generic outputs", async () => {
    let callback!: (payload: unknown) => void;
    const unsubscribe = vi.fn();
    const subscribe = vi.fn((_nodeId, _channel, cb) => {
      callback = cb;
      return unsubscribe;
    });

    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput: vi.fn(),
      subscribe,
    });

    const { result, rerender, unmount } = renderHook(
      ({ nodeId }) => useUIOutput<{ ok: boolean }>(nodeId, "data"),
      { initialProps: { nodeId: "node-1" as string | null } },
    );

    act(() => callback({ ok: true }));
    expect(result.current).toEqual({ ok: true });

    rerender({ nodeId: null });
    expect(unsubscribe).toHaveBeenCalled();
    expect(result.current).toBeNull();

    unmount();
  });

  it("keeps only string payloads for text output", async () => {
    let callback!: (payload: unknown) => void;
    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput: vi.fn(),
      subscribe: vi.fn((_nodeId, _channel, cb) => {
        callback = cb;
        return vi.fn();
      }),
    });

    const { result } = renderHook(() => useUITextOutput("node-1", "text"));
    act(() => callback({ nope: true }));
    expect(result.current).toBeNull();

    act(() => callback("hello"));
    expect(result.current).toBe("hello");

    const idle = renderHook(() => useUITextOutput(null, "text"));
    expect(idle.result.current).toBeNull();
  });

  it("creates and revokes object urls for video output", async () => {
    let callback!: (payload: unknown) => void;
    const revokeSpy = vi.spyOn(URL, "revokeObjectURL");
    const createSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:one")
      .mockReturnValueOnce("blob:two");

    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput: vi.fn(),
      subscribe: vi.fn((_nodeId, _channel, cb) => {
        callback = cb;
        return vi.fn();
      }),
    });

    const { result, unmount } = renderHook(() => useUIVideoOutput("node-1", "video"));
    act(() => callback("not-binary"));
    expect(result.current).toBeNull();

    act(() => callback(new Uint8Array([1, 2]).buffer));
    expect(result.current).toBe("blob:one");

    act(() => callback(new Uint8Array([3, 4]).buffer));
    expect(result.current).toBe("blob:two");
    expect(createSpy).toHaveBeenCalledTimes(2);
    expect(revokeSpy).toHaveBeenCalledWith("blob:one");

    unmount();
    expect(revokeSpy).toHaveBeenCalledWith("blob:two");

    const idle = renderHook(() => useUIVideoOutput(null, "video"));
    expect(idle.result.current).toBeNull();

    const unsubscribed = renderHook(() => useUIVideoOutput("node-2", "video"));
    unsubscribed.unmount();
  });
});


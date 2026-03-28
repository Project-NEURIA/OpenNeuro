import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSSE } from "./useSSE";

class MockEventSource {
  static instances: MockEventSource[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  close = vi.fn();

  constructor(public url: string) {
    MockEventSource.instances.push(this);
  }
}

describe("useSSE", () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    vi.stubGlobal("EventSource", MockEventSource as unknown as typeof EventSource);
  });

  it("tracks connection state, parses messages, ignores invalid json, and closes on cleanup", async () => {
    const { result, unmount } = renderHook(() => useSSE<{ ok: boolean }>("/metrics"));
    const source = MockEventSource.instances[0]!;

    act(() => {
      source.onopen?.();
    });
    await waitFor(() => expect(result.current.connected).toBe(true));

    act(() => {
      source.onmessage?.({ data: "{\"ok\":true}" } as MessageEvent<string>);
    });
    await waitFor(() => expect(result.current.data).toEqual({ ok: true }));

    act(() => {
      source.onmessage?.({ data: "not-json" } as MessageEvent<string>);
    });
    expect(result.current.data).toEqual({ ok: true });

    act(() => {
      source.onerror?.();
    });
    await waitFor(() => expect(result.current.connected).toBe(false));

    unmount();
    expect(source.close).toHaveBeenCalled();
  });
});

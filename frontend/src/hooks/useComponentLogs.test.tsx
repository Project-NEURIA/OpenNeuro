import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useComponentLogs } from "./useComponentLogs";
import * as api from "@/lib/api";

describe("useComponentLogs", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useFakeTimers();
  });

  it("resets when node is null", () => {
    const { result } = renderHook(() => useComponentLogs(null));
    expect(result.current.entries).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it("loads logs, appends entries, trims to 600, polls, and surfaces errors", async () => {
    const fetchLogs = vi.spyOn(api, "fetchComponentLogs");
    fetchLogs
      .mockResolvedValueOnce({
        node_id: "node-1",
        entries: Array.from({ length: 500 }, (_, index) => ({
          seq: index + 1,
          timestamp: index,
          stream: index % 2 === 0 ? "stdout" : "stderr",
          text: `line-${index + 1}`,
        })),
      })
      .mockResolvedValueOnce({
        node_id: "node-1",
        entries: Array.from({ length: 200 }, (_, index) => ({
          seq: index + 501,
          timestamp: index + 500,
          stream: "stdout",
          text: `line-${index + 501}`,
        })),
      })
      .mockResolvedValueOnce({
        node_id: "node-1",
        entries: [],
      })
      .mockRejectedValueOnce(new Error("broken"));

    const { result } = renderHook(() => useComponentLogs("node-1"));

    await act(async () => {});
    expect(result.current.entries).toHaveLength(500);
    expect(fetchLogs).toHaveBeenNthCalledWith(1, "node-1", 0, 300);

    await act(async () => {
      vi.advanceTimersByTime(500);
      await Promise.resolve();
    });
    expect(result.current.entries).toHaveLength(600);
    expect(result.current.entries[0]?.seq).toBe(101);

    await act(async () => {
      vi.advanceTimersByTime(500);
      await Promise.resolve();
    });
    expect(result.current.entries).toHaveLength(600);

    await act(async () => {
      vi.advanceTimersByTime(500);
      await Promise.resolve();
    });
    expect(result.current.error).toBe("broken");
  });

  it("stops updating after unmount", async () => {
    let resolveFetch!: (value: Awaited<ReturnType<typeof api.fetchComponentLogs>>) => void;
    vi.spyOn(api, "fetchComponentLogs").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveFetch = resolve;
        }),
    );

    const { result, unmount } = renderHook(() => useComponentLogs("node-2"));
    unmount();

    await act(async () => {
      resolveFetch({ node_id: "node-2", entries: [{ seq: 1, timestamp: 1, stream: "stdout", text: "late" }] });
    });

    expect(result.current.entries).toEqual([]);
  });

  it("ignores rejections that arrive after cancellation", async () => {
    let rejectFetch!: (reason?: unknown) => void;
    vi.spyOn(api, "fetchComponentLogs").mockImplementation(
      () =>
        new Promise((_, reject) => {
          rejectFetch = reject;
        }),
    );

    const { result, unmount } = renderHook(() => useComponentLogs("node-3"));
    unmount();

    await act(async () => {
      rejectFetch(new Error("late failure"));
    });

    expect(result.current.error).toBeNull();
  });

  it("falls back to a generic error message for non-Error rejections", async () => {
    vi.spyOn(api, "fetchComponentLogs").mockRejectedValue("bad");

    const { result } = renderHook(() => useComponentLogs("node-4"));
    await act(async () => {});

    expect(result.current.error).toBe("Failed to load logs");
  });
});

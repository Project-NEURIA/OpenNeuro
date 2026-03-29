import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useComponents } from "@/hooks/useComponents";
import { useGraphData } from "@/hooks/useGraphData";
import { useMetricsHistory } from "@/hooks/useMetricsHistory";
import * as api from "@/lib/api";
import * as sse from "@/hooks/useSSE";
import type { ComponentInfo, MetricsSnapshot } from "@/lib/types";

const baseComponent: ComponentInfo = {
  type_: "Mic",
  tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
  init: {},
  inputs: { input: "Audio" },
  outputs: { output: "Audio" },
  ui_inputs: {},
  ui_outputs: {},
};

function makeSnapshot(
  timestamp: number,
  nodeOverrides: Partial<MetricsSnapshot["nodes"]> = {},
): MetricsSnapshot {
  return {
    timestamp,
    nodes: {
      alpha: {
        name: "Mic",
        status: "running",
        senders: {
          out: {
            name: "out",
            msg_count_delta: 3,
            byte_count_delta: 300,
            last_send_time: timestamp,
            buffer_depth: 2,
          },
        },
        receivers: {
          in: {
            name: "in",
            msg_count_delta: 1,
            byte_count_delta: 100,
            lag: 4,
          },
        },
      },
      ...nodeOverrides,
    },
  };
}

describe("data hooks", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("loads components and logs fetch failures", async () => {
    vi.spyOn(api, "fetchComponents").mockResolvedValueOnce([baseComponent]);
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const success = renderHook(() => useComponents());
    await waitFor(() => expect(success.result.current).toEqual([baseComponent]));

    vi.spyOn(api, "fetchComponents").mockRejectedValueOnce(new Error("boom"));
    const failure = renderHook(() => useComponents());
    await waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith("[components] Fetch failed:", expect.any(Error));
      expect(failure.result.current).toEqual([]);
    });
  });

  it("maps components and forwards sse state", () => {
    vi.spyOn(sse, "useSSE").mockReturnValue({
      connected: true,
      data: makeSnapshot(10),
    });

    const { result } = renderHook(() => useGraphData([baseComponent]));
    expect(result.current.connected).toBe(true);
    expect(result.current.metrics?.timestamp).toBe(10);
    expect(result.current.componentMap).toEqual({ Mic: baseComponent });
  });

  it("accumulates metrics history and handles missing nodes, duplicate snapshots, and max length trimming", () => {
    const first = makeSnapshot(1);
    const second = makeSnapshot(2, { beta: undefined });
    const third = {
      timestamp: 3,
      nodes: {
        beta: {
          name: "Mic",
          status: "setup",
          senders: {},
          receivers: {},
        },
      },
    } satisfies MetricsSnapshot;

    const { result, rerender } = renderHook(
      ({ snapshot }) => useMetricsHistory(snapshot),
      { initialProps: { snapshot: null as MetricsSnapshot | null } },
    );

    expect(result.current.current).toBeNull();

    rerender({ snapshot: first });
    expect(result.current.current).toBe(first);
    expect(result.current.dt).toBe(0);
    expect(result.current.snapshotRate).toBe(0);

    rerender({ snapshot: first });
    expect(result.current.snapshots).toHaveLength(1);

    rerender({ snapshot: second });
    expect(result.current.dt).toBe(1);
    expect(result.current.nodeHistory.alpha.msgThroughput).toEqual([0, 3]);
    expect(result.current.nodeHistory.alpha.byteThroughput).toEqual([0, 300]);
    expect(result.current.nodeHistory.alpha.senderHistory.out.bufferDepths).toEqual([2, 2]);
    expect(result.current.nodeHistory.alpha.receiverHistory.in.msgThroughput).toEqual([0, 1]);

    rerender({
      snapshot: {
        timestamp: 2.5,
        nodes: {
          alpha: {
            name: "Mic",
            status: "running",
            senders: {},
            receivers: {},
          },
        },
      },
    });
    expect(result.current.nodeHistory.alpha.senderHistory.out.msgThroughput.at(-1)).toBe(0);
    expect(result.current.nodeHistory.alpha.receiverHistory.in.byteThroughput.at(-1)).toBe(0);

    rerender({ snapshot: third });
    expect(result.current.nodeHistory.alpha.msgThroughput.at(-1)).toBe(0);
    expect(result.current.nodeHistory.beta.msgThroughput.at(-1)).toBe(0);
    expect(result.current.snapshotRate).toBe(4 / 2);

    for (let i = 4; i <= 65; i++) {
      rerender({ snapshot: makeSnapshot(i) });
    }
    expect(result.current.snapshots).toHaveLength(60);
  });
});


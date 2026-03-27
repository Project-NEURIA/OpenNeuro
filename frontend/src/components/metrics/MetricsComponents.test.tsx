import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MetricsDashboard } from "./MetricsDashboard";
import { Waveform } from "./Waveform";
import type { MetricsHistory } from "@/hooks/useMetricsHistory";
import type { ComponentInfo } from "@/lib/types";

const componentMap: Record<string, ComponentInfo> = {
  SourceNode: {
    type_: "SourceNode",
    tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
    init: {},
    inputs: {},
    outputs: {},
    ui_inputs: {},
    ui_outputs: {},
  },
  SinkNode: {
    type_: "SinkNode",
    tags: { io: ["sink"], functionality: ["video"], gpu: ["cpu"] },
    init: {},
    inputs: {},
    outputs: {},
    ui_inputs: {},
    ui_outputs: {},
  },
};

const populatedHistory: MetricsHistory = {
  current: {
    timestamp: 2,
    nodes: {
      source: {
        name: "SourceNode",
        status: "running",
        senders: {
          out: {
            name: "out",
            msg_count_delta: 5,
            byte_count_delta: 1_024,
            last_send_time: 2,
            buffer_depth: 7,
          },
        },
        receivers: {},
      },
      sink: {
        name: "SinkNode",
        status: "stopped",
        senders: {},
        receivers: {
          in: {
            name: "in",
            msg_count_delta: 2,
            byte_count_delta: 512,
            lag: 6,
          },
        },
      },
    },
  },
  snapshots: [
    { timestamp: 1, nodes: {} },
    {
      timestamp: 2,
      nodes: {
        source: {
          name: "SourceNode",
          status: "running",
          senders: {
            out: {
              name: "out",
              msg_count_delta: 5,
              byte_count_delta: 1_024,
              last_send_time: 2,
              buffer_depth: 7,
            },
          },
          receivers: {},
        },
        sink: {
          name: "SinkNode",
          status: "stopped",
          senders: {},
          receivers: {
            in: {
              name: "in",
              msg_count_delta: 2,
              byte_count_delta: 512,
              lag: 6,
            },
          },
        },
      },
    },
  ],
  snapshotRate: 2,
  dt: 1,
  nodeHistory: {
    source: {
      msgThroughput: [0, 5],
      byteThroughput: [0, 1_024],
      senderHistory: {
        out: {
          msgThroughput: [0, 5],
          byteThroughput: [0, 1_024],
          bufferDepths: [0, 7],
        },
      },
      receiverHistory: {},
    },
    sink: {
      msgThroughput: [0, 0],
      byteThroughput: [0, 0],
      senderHistory: {},
      receiverHistory: {
        in: {
          msgThroughput: [0, 2],
          byteThroughput: [0, 512],
        },
      },
    },
  },
};

describe("metrics components", () => {
  it("renders dashboard totals, sorted node panels, sender and receiver sections", () => {
    const onClose = vi.fn();
    render(
      <MetricsDashboard
        connected
        history={populatedHistory}
        componentMap={componentMap}
        onClose={onClose}
      />,
    );

    expect(screen.getByText("Metrics")).toBeInTheDocument();
    expect(screen.getByText("Total Msg Throughput")).toBeInTheDocument();
    expect(screen.getAllByText("5/s").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1.0 KB/s").length).toBeGreaterThan(0);
    expect(screen.getByText("SourceNode")).toBeInTheDocument();
    expect(screen.getByText("SinkNode")).toBeInTheDocument();
    expect(screen.getByText("Output")).toBeInTheDocument();
    expect(screen.getByText("Input")).toBeInTheDocument();
    expect(screen.getByText("lag")).toBeInTheDocument();
    expect(screen.getByText("6")).toBeInTheDocument();
    expect(screen.getByText("source")).toBeInTheDocument();
    expect(screen.getByText("sink")).toBeInTheDocument();

    screen.getByRole("button").click();
    expect(onClose).toHaveBeenCalled();
  });

  it("renders empty dashboard state and waveform variants", () => {
    render(
      <>
        <MetricsDashboard
          connected={false}
          history={{
            current: null,
            snapshots: [],
            snapshotRate: 0,
            dt: 0,
            nodeHistory: {},
          }}
          componentMap={{}}
          onClose={() => {}}
        />
        <Waveform data={[]} />
        <Waveform data={[0, 5, 10]} showAxes duration={10} formatY={(value) => `${value}x`} />
      </>,
    );

    expect(screen.getByText("Awaiting graph data")).toBeInTheDocument();
    expect(screen.getByText("-10s")).toBeInTheDocument();
    expect(screen.getByText("now")).toBeInTheDocument();
    expect(screen.getByText("10x")).toBeInTheDocument();
  });
});

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  getMetricsCategoryOrder,
  MetricsDashboard,
  sortMetricsNodeIds,
} from "@/components/metrics/MetricsDashboard";
import { NodePanel } from "@/components/metrics/NodePanel";
import { ReceiverSection } from "@/components/metrics/ReceiverSection";
import { SenderSection } from "@/components/metrics/SenderSection";
import { SystemTotals } from "@/components/metrics/SystemTotals";
import { Waveform } from "@/components/metrics/Waveform";
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
  it("covers the metrics sorting helper directly", () => {
    expect(getMetricsCategoryOrder("source")).toBe(0);
    expect(getMetricsCategoryOrder("mystery")).toBe(1);

    expect(
      sortMetricsNodeIds(
        {
          zebra: { name: "ZebraNode", status: "running", senders: {}, receivers: {} },
          alpha: { name: "AlphaNode", status: "running", senders: {}, receivers: {} },
        },
        {},
      ),
    ).toEqual(["alpha", "zebra"]);

    expect(
      sortMetricsNodeIds(
        {
          sink: { name: "SinkNode", status: "running", senders: {}, receivers: {} },
          source: { name: "SourceNode", status: "running", senders: {}, receivers: {} },
        },
        componentMap,
      ),
    ).toEqual(["source", "sink"]);
  });

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

  it("sorts nodes alphabetically when categories match", () => {
    render(
      <MetricsDashboard
        connected
        history={{
          current: {
            timestamp: 1,
            nodes: {
              b: { name: "SourceNode", status: "running", senders: {}, receivers: {} },
              a: { name: "AnotherSource", status: "running", senders: {}, receivers: {} },
            },
          },
          snapshots: [{ timestamp: 1, nodes: {} }],
          snapshotRate: 1,
          dt: 1,
          nodeHistory: {
            a: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
            b: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
          },
        }}
        componentMap={{
          ...componentMap,
          AnotherSource: {
            type_: "AnotherSource",
            tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
            init: {},
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
        }}
        onClose={() => {}}
      />,
    );

    const labels = screen.getAllByText(/Source/).map((node) => node.textContent);
    expect(labels.indexOf("AnotherSource")).toBeLessThan(labels.indexOf("SourceNode"));
  });

  it("covers fallback branches in metric subcomponents", () => {
    render(
      <>
        <MetricsDashboard
          connected
          history={{
            current: {
              timestamp: 1,
              nodes: {
                sink: { name: "SinkNode", status: "running", senders: {}, receivers: {} },
                unknown: { name: "MissingNode", status: "startup", senders: {}, receivers: {} },
              },
            },
            snapshots: [{ timestamp: 1, nodes: {} }, { timestamp: 4, nodes: {} }],
            snapshotRate: 1,
            dt: 1,
            nodeHistory: {},
          }}
          componentMap={{
            ...componentMap,
            SinkNode: {
              ...componentMap.SinkNode!,
              tags: { io: ["weird" as never], functionality: ["video"], gpu: ["cpu"] },
            },
          }}
          onClose={() => {}}
        />
        <NodePanel
          nodeId="unknown"
          metrics={{ name: "MissingNode", status: "mystery", senders: {}, receivers: {} }}
          history={{ msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} }}
          dt={0}
          duration={0}
          componentMap={{
            MissingNode: {
              type_: "MissingNode",
              tags: { io: [] as never[], functionality: ["misc"], gpu: ["cpu"] },
              init: {},
              inputs: {},
              outputs: {},
              ui_inputs: {},
              ui_outputs: {},
            },
          }}
          allNodes={{}}
        />
        <ReceiverSection
          receiver={{ name: "idle", msg_count_delta: 2, byte_count_delta: 128, lag: 0 }}
          dt={0}
          duration={2}
        />
        <ReceiverSection
          receiver={{ name: "warm", msg_count_delta: 2, byte_count_delta: 128, lag: 3 }}
          dt={1}
          duration={2}
          receiverHistory={{ msgThroughput: [1], byteThroughput: [2] }}
        />
        <SenderSection
          sender={{ name: "sender", msg_count_delta: 0, byte_count_delta: 0, last_send_time: 0, buffer_depth: 0 }}
          dt={0}
          duration={2}
        />
        <SystemTotals
          history={{
            current: null,
            snapshots: [{ timestamp: 1, nodes: {} }, { timestamp: 2, nodes: {} }],
            snapshotRate: 1,
            dt: 1,
            nodeHistory: {
              a: { msgThroughput: [1, 2], byteThroughput: [3, 4], senderHistory: {}, receiverHistory: {} },
              b: { msgThroughput: [5], byteThroughput: [6], senderHistory: {}, receiverHistory: {} },
            },
          }}
        />
        <Waveform data={[5]} showAxes />
      </>,
    );

    expect(screen.getAllByText("No channels").length).toBeGreaterThan(0);
    expect(screen.getByText("idle")).toBeInTheDocument();
    expect(screen.getByText("warm")).toBeInTheDocument();
    expect(screen.getByText("sender")).toBeInTheDocument();
    expect(screen.getAllByText("0").length).toBeGreaterThan(0);
  });

  it("falls back to conduit ordering when component categories are unknown or missing", () => {
    render(
      <MetricsDashboard
        connected
        history={{
          current: {
            timestamp: 3,
            nodes: {
              zed: { name: "ZedNode", status: "running", senders: {}, receivers: {} },
              alpha: { name: "AlphaNode", status: "running", senders: {}, receivers: {} },
            },
          },
          snapshots: [{ timestamp: 3, nodes: {} }],
          snapshotRate: 1,
          dt: 1,
          nodeHistory: {
            zed: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
            alpha: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
          },
        }}
        componentMap={{
          ZedNode: {
            type_: "ZedNode",
            tags: { io: ["weird" as never], functionality: ["misc"], gpu: ["cpu"] },
            init: {},
            inputs: {},
            outputs: {},
            ui_inputs: {},
            ui_outputs: {},
          },
        }}
        onClose={() => {}}
      />,
    );

    const labels = screen.getAllByText(/Node$/).map((node) => node.textContent);
    expect(labels.indexOf("AlphaNode")).toBeLessThan(labels.indexOf("ZedNode"));
  });

  it("falls back for the compared node category when the second item is missing", () => {
    render(
      <MetricsDashboard
        connected
        history={{
          current: {
            timestamp: 4,
            nodes: {
              source: { name: "SourceNode", status: "running", senders: {}, receivers: {} },
              missing: { name: "MissingNode", status: "running", senders: {}, receivers: {} },
            },
          },
          snapshots: [{ timestamp: 4, nodes: {} }],
          snapshotRate: 1,
          dt: 1,
          nodeHistory: {
            source: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
            missing: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
          },
        }}
        componentMap={componentMap}
        onClose={() => {}}
      />,
    );

    const labels = screen.getAllByText(/Node$/).map((node) => node.textContent);
    expect(labels.indexOf("SourceNode")).toBeLessThan(labels.indexOf("MissingNode"));
  });

  it("falls back to conduit ordering when both compared categories are missing", () => {
    render(
      <MetricsDashboard
        connected
        history={{
          current: {
            timestamp: 5,
            nodes: {
              zebra: { name: "ZebraNode", status: "running", senders: {}, receivers: {} },
              alpha: { name: "AlphaNode", status: "running", senders: {}, receivers: {} },
            },
          },
          snapshots: [{ timestamp: 5, nodes: {} }],
          snapshotRate: 1,
          dt: 1,
          nodeHistory: {
            zebra: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
            alpha: { msgThroughput: [], byteThroughput: [], senderHistory: {}, receiverHistory: {} },
          },
        }}
        componentMap={{}}
        onClose={() => {}}
      />,
    );

    const labels = screen.getAllByText(/Node$/).map((node) => node.textContent);
    expect(labels.indexOf("AlphaNode")).toBeLessThan(labels.indexOf("ZebraNode"));
  });
});


import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "@/App";
import * as api from "@/lib/api";
import * as graphDataHook from "@/hooks/useGraphData";
import * as componentsHook from "@/hooks/useComponents";
import * as metricsHistoryHook from "@/hooks/useMetricsHistory";
import * as layout from "@/lib/layout";
import * as typecheck from "@/lib/typecheck";
import type { MetricsSnapshot } from "@/lib/types";

let currentGraphCanvasProps: Record<string, unknown> | null = null;
let cachedEditConfig: (() => void) | undefined;
type MockGraphNode = {
  id: string;
  type?: string;
  data?: Record<string, unknown>;
};

type MockGraphEdge = {
  id: string;
};

let currentMetrics: MetricsSnapshot = {
  timestamp: 1,
  nodes: {
    n1: { name: "Mic", status: "running", senders: {}, receivers: {} },
    n2: {
      name: "Speaker",
      status: "stopped",
      senders: {},
      receivers: { input: { name: "input", msg_count_delta: 1, byte_count_delta: 64, lag: 0 } },
    },
  },
};

vi.mock("@xyflow/react", async () => {
  const ReactModule = await import("react");
  return {
    ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    useNodesState: (initial: MockGraphNode[]) => {
      const [state, setState] = ReactModule.useState<MockGraphNode[]>(initial);
      return [
        state,
        setState,
        (changes: { type: string; id: string }[]) => {
          setState((prev) => prev.filter((node) => !changes.some((change) => change.type === "remove" && change.id === node.id)));
        },
      ];
    },
    useEdgesState: (initial: MockGraphEdge[]) => {
      const [state, setState] = ReactModule.useState<MockGraphEdge[]>(initial);
      return [
        state,
        setState,
        (changes: { type: string; id: string }[]) => {
          setState((prev) => prev.filter((edge) => !changes.some((change) => change.type === "remove" && change.id === edge.id)));
        },
      ];
    },
    addEdge: (edge: Record<string, unknown>, edges: Record<string, unknown>[]) => [
      ...edges,
      { id: "connected-edge", ...edge },
    ],
    useReactFlow: () => ({
      screenToFlowPosition: ({ x, y }: { x: number; y: number }) => ({ x: x + 1, y: y + 1 }),
    }),
  };
});

vi.mock("@/contexts/UIChannelContext", () => ({
  UIChannelProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/components/EnvEditor", () => ({
  EnvEditor: ({ onClose }: { onClose: () => void }) => (
    <button onClick={onClose}>close-env</button>
  ),
}));

vi.mock("@/components/SplashScreen", () => ({
  SplashScreen: ({ status }: { status: string }) => <div>{status}</div>,
}));

vi.mock("@/components/graph/GraphCanvas", () => ({
  GraphCanvas: (props: Record<string, unknown>) => {
    currentGraphCanvasProps = props;
    const nodes = (props.nodes as MockGraphNode[]) ?? [];
    const edges = (props.edges as MockGraphEdge[]) ?? [];
    const firstGraphNode = nodes.find((node) => node.type === "graph");
    const ghostGraphNode = nodes.find((node) => node.type === "graph" && node.data?.label === "Ghost");
    const compositeNode = nodes.find((node) => node.type === "graph" && node.data?.category === "composite");
    const configuringNode = nodes.find((node) => node.type === "configuring");
    const firstEdge = edges[0];
    const firstGraphNodeEdit = firstGraphNode?.data?.onEditConfig as (() => void) | undefined;
    const ghostGraphNodeEdit = ghostGraphNode?.data?.onEditConfig as (() => void) | undefined;
    const configuringConfirm = configuringNode?.data?.onConfirm as ((config: { gain: number }) => void) | undefined;
    const configuringCancel = configuringNode?.data?.onCancel as (() => void) | undefined;

    const makeDropEvent = (rawApplication: string, rawText = "") => ({
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      clientX: 20,
      clientY: 30,
      dataTransfer: {
        dropEffect: "",
        getData: (type: string) =>
          type === "application/openneuro" ? rawApplication : rawText,
      },
    });

    return (
      <div>
        <div data-testid="canvas-nodes">
          {nodes.map((node) => `${node.id}:${node.type}:${String(node.data?.label ?? "")}`).join("|")}
        </div>
        <div data-testid="canvas-edges">
          {edges.map((edge) => edge.id).join("|")}
        </div>
        <button onClick={() => (props.onSelectionChange as ((payload: { nodes: unknown[] }) => void) | undefined)?.({ nodes: nodes.filter((node) => node.type === "graph") })}>
          select-all
        </button>
        <button onClick={() => firstGraphNode && (props.onSelectionChange as ((payload: { nodes: unknown[] }) => void) | undefined)?.({ nodes: [firstGraphNode] })}>
          select-first
        </button>
        <button
          onClick={() =>
            firstGraphNode &&
            (props.onNodeContextMenu as ((event: React.MouseEvent, node: unknown) => void) | undefined)?.(
              { preventDefault: vi.fn(), clientX: 10, clientY: 15 } as never,
              firstGraphNode,
            )
          }
        >
          context-first
        </button>
        <button
          onClick={() =>
            compositeNode &&
            (props.onNodeContextMenu as ((event: React.MouseEvent, node: unknown) => void) | undefined)?.(
              { preventDefault: vi.fn(), clientX: 10, clientY: 15 } as never,
              compositeNode,
            )
          }
        >
          context-composite
        </button>
        <button onClick={() => firstGraphNodeEdit?.()}>edit-first</button>
        <button onClick={() => ghostGraphNodeEdit?.()}>edit-ghost</button>
        <button onClick={() => {
          cachedEditConfig = firstGraphNodeEdit;
        }}>
          cache-edit-first
        </button>
        <button onClick={() => cachedEditConfig?.()}>edit-cached</button>
        <button onClick={() => configuringConfirm?.({ gain: 5 })}>confirm-configuring</button>
        <button onClick={() => configuringCancel?.()}>cancel-configuring</button>
        <button onClick={() => firstGraphNode && (props.onNodesChange as ((changes: unknown[]) => void) | undefined)?.([{ type: "remove", id: firstGraphNode.id }])}>
          remove-node
        </button>
        <button onClick={() => firstEdge && (props.onEdgesChange as ((changes: unknown[]) => void) | undefined)?.([{ type: "remove", id: firstEdge.id }])}>
          remove-edge
        </button>
        <button
          onClick={() =>
            (props.onConnect as ((connection: Record<string, unknown>) => void) | undefined)?.({
              source: "n1",
              sourceHandle: "out-output",
              target: "n2",
              targetHandle: "in-input",
            })
          }
        >
          connect
        </button>
        <button
          onClick={() =>
            (props.onNodeDragStop as ((event: unknown, node: unknown) => void) | undefined)?.(
              {},
              { id: "n1", position: { x: 99, y: 101 } },
            )
          }
        >
          drag-stop
        </button>
        <button
          onClick={() =>
            (props.onDrop as ((event: React.DragEvent) => void) | undefined)?.(
              makeDropEvent(
                JSON.stringify({
                  kind: "component",
                  type_: "Speaker",
                  tags: { io: ["sink"], functionality: ["audio"], gpu: ["cpu"] },
                  init: { simple: { type: "string" } },
                  inputs: { input: "Audio" },
                  outputs: {},
                  ui_inputs: {},
                  ui_outputs: {},
                }),
              ) as never,
            )
          }
        >
          drop-simple
        </button>
        <button
          onClick={() =>
            (props.onDrop as ((event: React.DragEvent) => void) | undefined)?.(
              makeDropEvent(
                JSON.stringify({
                  kind: "component",
                  type_: "Mic",
                  tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
                  init: {
                    config: { type: "object", properties: { gain: { type: "number" } } },
                  },
                  inputs: {},
                  outputs: { output: "Audio" },
                  ui_inputs: {},
                  ui_outputs: {},
                }),
              ) as never,
            )
          }
        >
          drop-config
        </button>
        <button
          onClick={() =>
            (props.onDrop as ((event: React.DragEvent) => void) | undefined)?.(
              makeDropEvent(JSON.stringify({ kind: "project", name: "ProjectComposite" })) as never,
            )
          }
        >
          drop-project
        </button>
        <button onClick={() => (props.onPaneClick as (() => void) | undefined)?.()}>pane-click</button>
      </div>
    );
  },
}));

vi.mock("@/components/graph/NodeSidebar", () => ({
  NodeSidebar: ({
    components,
    currentProject,
  }: {
    components: unknown[];
    currentProject: string;
  }) => <div>{`sidebar:${components.length}:${currentProject}`}</div>,
}));

vi.mock("@/components/graph/MetricsOverlay", () => ({
  MetricsOverlay: ({
    connected,
    loggingOpen,
    onToggleLogging,
    onOpenDashboard,
    onOpenEnv,
  }: {
    connected: boolean;
    loggingOpen: boolean;
    onToggleLogging: () => void;
    onOpenDashboard: () => void;
    onOpenEnv: () => void;
  }) => (
    <div>
      <div>{`overlay:${connected}:${loggingOpen}`}</div>
      <button onClick={onToggleLogging}>toggle-logging</button>
      <button onClick={onOpenDashboard}>open-dashboard</button>
      <button onClick={onOpenEnv}>open-env</button>
    </div>
  ),
}));

vi.mock("@/components/graph/LoggingPanel", () => ({
  LoggingPanel: ({ selectedNode }: { selectedNode: { id: string } | null }) => (
    <div>{`logging:${selectedNode?.id ?? "none"}`}</div>
  ),
}));

vi.mock("@/components/metrics/MetricsDashboard", () => ({
  MetricsDashboard: ({ onClose }: { onClose: () => void }) => (
    <button onClick={onClose}>close-dashboard</button>
  ),
}));

vi.mock("@/components/project/ProjectChooser", () => ({
  ProjectChooser: ({
    hadProject,
    onOpen,
    onCancel,
  }: {
    hadProject: boolean;
    onOpen: (name: string) => void;
    onCancel: () => void;
  }) => (
    <div>
      <div>{`chooser:${hadProject}`}</div>
      <button onClick={() => onOpen("OpenedProject")}>open-project</button>
      <button onClick={onCancel}>cancel-project</button>
    </div>
  ),
}));

const components = [
  {
    type_: "Mic",
    tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
    init: { config: { type: "object", properties: { gain: { type: "number" } } } },
    inputs: {},
    outputs: { output: "Sender[Audio]" },
    ui_inputs: {},
    ui_outputs: {},
  },
  {
    type_: "Speaker",
    tags: { io: ["sink"], functionality: ["audio"], gpu: ["cpu"] },
    init: { simple: { type: "string" } },
    inputs: { input: "Receiver[Audio]" },
    outputs: {},
    ui_inputs: {},
    ui_outputs: {},
  },
];

const componentMap = {
  Mic: components[0]!,
  Speaker: components[1]!,
};

function getCanvasProps() {
  expect(currentGraphCanvasProps).not.toBeNull();
  return currentGraphCanvasProps as {
    nodes: Array<Record<string, unknown>>;
    edges: Array<Record<string, unknown>>;
    onNodesChange?: (changes: { type: string; id: string }[]) => void;
    onEdgesChange?: (changes: { type: string; id: string }[]) => void;
    onConnect?: (connection: {
      source?: string | null;
      target?: string | null;
      sourceHandle?: string | null;
      targetHandle?: string | null;
    }) => void;
    onDragOver?: (event: React.DragEvent) => void;
    onDrop?: (event: React.DragEvent) => void;
  };
}

describe("App", () => {
  beforeEach(() => {
    currentGraphCanvasProps = null;
    cachedEditConfig = undefined;
    currentMetrics = {
      timestamp: 1,
      nodes: {
        n1: { name: "Mic", status: "running", senders: {}, receivers: {} },
        n2: {
          name: "Speaker",
          status: "stopped",
          senders: {},
          receivers: { input: { name: "input", msg_count_delta: 1, byte_count_delta: 64, lag: 0 } },
        },
      },
    };
    vi.restoreAllMocks();

    vi.spyOn(componentsHook, "useComponents").mockReturnValue(components as never);
    vi.spyOn(graphDataHook, "useGraphData").mockImplementation(() => ({
      connected: true,
      metrics: currentMetrics,
      componentMap: componentMap as never,
    }));
    vi.spyOn(metricsHistoryHook, "useMetricsHistory").mockReturnValue({
      current: null,
      snapshots: [],
      snapshotRate: 0,
      dt: 0,
      nodeHistory: {},
    });

    vi.spyOn(layout, "layoutNodes").mockReturnValue([
      { id: "n1", x: 5, y: 6 },
      { id: "n2", x: 7, y: 8 },
    ]);

    vi.spyOn(typecheck, "collectLeafNames").mockImplementation((name) => [name]);
    vi.spyOn(typecheck, "warmSubtypeCache").mockResolvedValue();
    vi.spyOn(typecheck, "typeToString").mockImplementation((type) => (type as { name?: string }).name ?? "unknown");
    vi.spyOn(typecheck, "checkTypes").mockReturnValue({
      types: new Map([
        ["n1.out.output", { kind: "concrete", name: "Audio" }],
        ["n2.in.input", { kind: "concrete", name: "Audio" }],
      ]),
      errors: [],
    });

    vi.spyOn(api, "fetchProjects").mockResolvedValue([]);
    vi.spyOn(api, "fetchNodes").mockResolvedValue([
      {
        id: "n1",
        type: "Mic",
        is_composite: false,
        status: "startup",
        x: 0,
        y: 0,
        init_args: {},
      },
      {
        id: "n2",
        type: "Speaker",
        is_composite: false,
        status: "stopped",
        x: 0,
        y: 0,
        init_args: {},
      },
    ]);
    vi.spyOn(api, "fetchEdges").mockResolvedValue([
      {
        source_node: "n1",
        source_slot: "output",
        target_node: "n2",
        target_slot: "input",
      },
    ]);
    vi.spyOn(api, "createNode").mockImplementation(async (type, initArgs) => ({
      id: type === "ProjectComposite" ? "composite-1" : `${type}-node`,
      type,
      is_composite: type === "ProjectComposite",
      status: "running",
      x: 0,
      y: 0,
      init_args: initArgs ?? {},
      inputs: type === "ProjectComposite" ? { "group.input": "Receiver[Audio]" } : undefined,
      outputs: type === "ProjectComposite" ? { "group.output": "Sender[Audio]" } : undefined,
    }));
    vi.spyOn(api, "updateNode").mockResolvedValue();
    vi.spyOn(api, "updateNodeInitArgs").mockResolvedValue({
      id: "n1",
      type: "Mic",
      status: "running",
      x: 0,
      y: 0,
      init_args: { gain: 5 },
    });
    vi.spyOn(api, "deleteNode").mockResolvedValue();
    vi.spyOn(api, "createEdge").mockResolvedValue({
      source_node: "n1",
      source_slot: "output",
      target_node: "n2",
      target_slot: "input",
    });
    vi.spyOn(api, "deleteEdge").mockResolvedValue();
    vi.spyOn(api, "createSubgraph").mockResolvedValue({
      id: "subgraph",
      type: "ProjectComposite",
      is_composite: true,
      status: "running",
      x: 0,
      y: 0,
    });
    vi.spyOn(api, "ungroupNode").mockResolvedValue([]);
    vi.spyOn(api, "fetchIsType").mockResolvedValue(true);
    vi.spyOn(api, "saveGraph").mockResolvedValue();
    vi.spyOn(api, "closeProject").mockResolvedValue();
    vi.spyOn(api, "fetchCurrentProject").mockResolvedValue({ current_project: null });
    vi.spyOn(api, "startProject").mockResolvedValue({ current_project: "Alpha" });
  });

  it("opens the chooser when there is no current project and enters the editor", async () => {
    render(<App />);

    expect(screen.getByText("Connecting to backend...")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());

    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByText("sidebar:2:OpenedProject")).toBeInTheDocument());
    expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic");
  });

  it("builds fallback and composite graph nodes through app initialization", async () => {
    vi.spyOn(api, "fetchCurrentProject").mockResolvedValueOnce({ current_project: "Alpha" });
    vi.spyOn(api, "fetchNodes").mockResolvedValueOnce([
      {
        id: "ghost",
        type: "Ghost",
        is_composite: false,
        status: "ready",
        x: 11,
        y: 12,
        init_args: undefined,
      },
      {
        id: "group",
        type: "ProjectComposite",
        is_composite: true,
        status: "running",
        x: 21,
        y: 22,
        init_args: {},
        inputs: { "group.input-main": "Receiver[Audio]" },
        outputs: { "group.output-main": "Sender[Audio]" },
      },
    ]);
    vi.spyOn(api, "fetchEdges").mockResolvedValueOnce([
      {
        source_node: "ghost",
        source_slot: "audio-main",
        target_node: "group",
        target_slot: "input-main",
      },
    ]);

    render(<App />);

    await waitFor(() => expect(screen.getByText("sidebar:2:Alpha")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("ghost:graph:Ghost"));
    expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("group:graph:ProjectComposite");
    expect(screen.getByTestId("canvas-edges")).toHaveTextContent("ghost:audio-main->group:input-main");

    const props = getCanvasProps();
    const ghostNode = props.nodes.find((node) => node.id === "ghost") as {
      position: { x: number; y: number };
      data: { category: string; inputs: string[]; outputs: string[] };
    };
    const compositeNode = props.nodes.find((node) => node.id === "group") as {
      position: { x: number; y: number };
      data: { category: string; inputs: string[]; outputs: string[] };
    };

    expect(ghostNode.position).toEqual({ x: 11, y: 12 });
    expect(ghostNode.data.category).toBe("conduit");
    expect(ghostNode.data.inputs).toEqual([]);
    expect(ghostNode.data.outputs).toEqual([]);

    expect(compositeNode.position).toEqual({ x: 21, y: 22 });
    expect(compositeNode.data.category).toBe("composite");
    expect(compositeNode.data.inputs).toEqual(["group.input-main"]);
    expect(compositeNode.data.outputs).toEqual(["group.output-main"]);
  });

  it("retries backend polling and falls back to the chooser when auto-start fails", async () => {
    vi.useFakeTimers();
    vi.spyOn(api, "fetchCurrentProject")
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ current_project: "Alpha" });
    vi.spyOn(api, "startProject").mockRejectedValueOnce(new Error("boom"));

    render(<App />);
    expect(screen.getByText("Connecting to backend...")).toBeInTheDocument();

    await act(async () => {
      await Promise.resolve();
      vi.runOnlyPendingTimers();
      await Promise.resolve();
    });
    vi.useRealTimers();
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
  });

  it("auto-starts into the editor when the backend already has a current project", async () => {
    vi.spyOn(api, "fetchCurrentProject").mockResolvedValueOnce({ current_project: "Alpha" });

    render(<App />);

    await waitFor(() => expect(screen.getByText("sidebar:2:Alpha")).toBeInTheDocument());
    expect(api.startProject).toHaveBeenCalledWith("Alpha");
  });

  it("skips concrete-type warming when the component registry exposes no typed slots", async () => {
    vi.spyOn(componentsHook, "useComponents").mockReturnValue([
      {
        type_: "Bare",
        tags: { io: ["conduit"], functionality: ["misc"], gpu: ["cpu"] },
        init: {},
        inputs: {},
        outputs: {},
        ui_inputs: {},
        ui_outputs: {},
      },
    ] as never);
    vi.spyOn(graphDataHook, "useGraphData").mockImplementation(() => ({
      connected: true,
      metrics: null,
      componentMap: {} as never,
    }));

    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByText("sidebar:1:OpenedProject")).toBeInTheDocument());
    expect(api.fetchIsType).not.toHaveBeenCalled();
    expect(typecheck.warmSubtypeCache).not.toHaveBeenCalled();
  });

  it("handles editor interactions for logging, config, graph edits, grouping, and home navigation", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));

    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic"));

    fireEvent.click(screen.getByText("toggle-logging"));
    expect(screen.getByText("logging:none")).toBeInTheDocument();
    fireEvent.click(screen.getByText("select-first"));
    expect(screen.getByText("logging:n1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("open-dashboard"));
    expect(screen.getByText("close-dashboard")).toBeInTheDocument();
    fireEvent.click(screen.getByText("close-dashboard"));
    fireEvent.click(screen.getByText("open-env"));
    expect(screen.getByText("close-env")).toBeInTheDocument();
    fireEvent.click(screen.getByText("close-env"));

    fireEvent.click(screen.getByText("connect"));
    fireEvent.click(screen.getByText("drag-stop"));
    fireEvent.click(screen.getByText("remove-edge"));
    await waitFor(() => expect(api.createEdge).toHaveBeenCalled());
    expect(api.updateNode).toHaveBeenCalledWith("n1", { x: 99, y: 101 });
    expect(api.deleteEdge).toHaveBeenCalled();

    fireEvent.click(screen.getByText("drop-config"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring"));
    fireEvent.click(screen.getByText("confirm-configuring"));
    await waitFor(() => expect(api.createNode).toHaveBeenCalledWith("Mic", { gain: 5 }));

    fireEvent.click(screen.getByText("edit-first"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-edit-n1-"));
    fireEvent.click(screen.getByText("cancel-configuring"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).not.toHaveTextContent("configuring-edit-n1-"));

    fireEvent.click(screen.getByText("drop-simple"));
    await waitFor(() => expect(api.createNode).toHaveBeenCalledWith("Speaker", undefined));

    fireEvent.click(screen.getByText("remove-node"));
    await waitFor(() => expect(api.deleteNode).toHaveBeenCalledWith("n1"));

    fireEvent.click(screen.getByText("drop-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("composite-1:graph:ProjectComposite"));
    fireEvent.click(screen.getByText("context-composite"));
    fireEvent.click(screen.getByText("Ungroup subgraph"));
    await waitFor(() => expect(api.ungroupNode).toHaveBeenCalledWith("composite-1"));

    fireEvent.click(screen.getByText("select-all"));
    fireEvent.click(screen.getByText("context-first"));
    fireEvent.click(screen.getByText("Group into subgraph"));
    fireEvent.change(screen.getByDisplayValue("Subgraph"), { target: { value: "Grouped" } });
    fireEvent.click(screen.getByText("Create"));
    await waitFor(() => expect(api.createSubgraph).toHaveBeenCalledWith(expect.any(Array), "Grouped"));
    fireEvent.click(screen.getByText("pane-click"));

    fireEvent.click(screen.getByTitle("Back to projects"));
    await waitFor(() => expect(api.closeProject).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
  });

  it("covers editor fallback branches for missing types, metrics, and drag events", async () => {
    currentMetrics = {
      timestamp: 2,
      nodes: {
        n1: { name: "Mic", status: "running", senders: {}, receivers: {} },
      },
    };
    vi.spyOn(api, "fetchNodes").mockResolvedValueOnce([
      {
        id: "n1",
        type: "Mic",
        is_composite: false,
        status: "startup",
        x: 11,
        y: 12,
        init_args: { gain: 1 },
      },
      {
        id: "ghost",
        type: "Ghost",
        is_composite: false,
        status: "idle",
        x: 40,
        y: 55,
        init_args: {},
      },
      {
        id: "n2",
        type: "Speaker",
        is_composite: false,
        status: "stopped",
        x: 70,
        y: 80,
        init_args: {},
      },
    ]);
    vi.spyOn(typecheck, "checkTypes").mockReturnValue({
      types: new Map([
        ["n1.out.output", { kind: "concrete", name: "Audio" }],
        ["n2.in.input", { kind: "concrete", name: "Audio" }],
      ]),
      errors: [
        {
          left: { kind: "concrete", name: "Audio" },
          right: { kind: "concrete", name: "Video" },
          constraint: {
            left: { kind: "concrete", name: "Audio" },
            right: { kind: "concrete", name: "Video" },
            origin: {
              kind: "edge",
              sourceNode: "n1",
              sourceSlot: "output",
              targetNode: "n2",
              targetSlot: "input",
            },
          },
        },
      ],
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("ghost:graph:Ghost"));

    expect(layout.layoutNodes).not.toHaveBeenCalled();

    const ghostNode = getCanvasProps().nodes.find((node) => node.id === "ghost") as {
      position: { x: number; y: number };
      data: { category: string; inputs: string[]; outputs: string[] };
    };

    expect(ghostNode.position).toEqual({ x: 40, y: 55 });
    expect(ghostNode.data.category).toBe("conduit");
    expect(ghostNode.data.inputs).toEqual([]);
    expect(ghostNode.data.outputs).toEqual([]);

    currentMetrics = {
      timestamp: 3,
      nodes: {
        n1: { name: "Mic", status: "running", senders: {}, receivers: {} },
      },
    };
    fireEvent.click(screen.getByText("toggle-logging"));

    await waitFor(() => {
      const props = getCanvasProps();
      const speakerNode = props.nodes.find((node) => node.id === "n2") as {
        data: { status: string; nodeMetrics: unknown };
      };
      const typedEdge = props.edges[0] as { data: { typeError?: string; byteDelta?: number } };

      expect(speakerNode.data.status).toBe("stopped");
      expect(speakerNode.data.nodeMetrics).toBeNull();
      expect(typedEdge.data.typeError).toBe("Audio ≠ Video");
      expect(typedEdge.data.byteDelta ?? 0).toBe(0);
    });

    const dragEvent = {
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
      dataTransfer: { dropEffect: "copy" },
    } as unknown as React.DragEvent;
    act(() => {
      getCanvasProps().onDragOver?.(dragEvent);
    });
    expect(dragEvent.preventDefault).toHaveBeenCalled();
    expect(dragEvent.stopPropagation).toHaveBeenCalled();
    expect(dragEvent.dataTransfer.dropEffect).toBe("move");

    const nodesBeforeInvalidDrop = screen.getByTestId("canvas-nodes").textContent;
    await act(async () => {
      getCanvasProps().onDrop?.({
        preventDefault: vi.fn(),
        clientX: 1,
        clientY: 2,
        dataTransfer: {
          getData: (type: string) => (type === "application/openneuro" ? "{bad-json" : ""),
        },
      } as unknown as React.DragEvent);
    });
    expect(screen.getByTestId("canvas-nodes").textContent).toBe(nodesBeforeInvalidDrop);

    fireEvent.click(screen.getByText("edit-ghost"));
    expect(screen.getByTestId("canvas-nodes")).not.toHaveTextContent("configuring-edit-ghost");

    fireEvent.click(screen.getByText("cache-edit-first"));
    fireEvent.click(screen.getByText("remove-node"));
    await waitFor(() => expect(api.deleteNode).toHaveBeenCalledWith("n1"));
    fireEvent.click(screen.getByText("edit-cached"));
    expect(screen.getByTestId("canvas-nodes")).not.toHaveTextContent("configuring-edit-n1-");
  });

  it("covers resolved-type reuse, missing layout positions, and config-schema fallbacks", async () => {
    vi.spyOn(api, "fetchNodes").mockResolvedValueOnce([
      {
        id: "n1",
        type: "Mic",
        is_composite: false,
        status: "startup",
        x: 0,
        y: 0,
        init_args: undefined,
      },
      {
        id: "n2",
        type: "Speaker",
        is_composite: false,
        status: "stopped",
        x: 0,
        y: 0,
        init_args: {},
      },
    ]);
    vi.spyOn(layout, "layoutNodes").mockReturnValue([{ id: "n1", x: 15, y: 25 }]);
    vi.spyOn(typecheck, "checkTypes").mockReturnValue({
      types: new Map([
        ["n1.in.input", { kind: "var", name: "n1.T" }],
        ["n2.in.left", { kind: "concrete", name: "Audio" }],
        ["n2.out.right", { kind: "concrete", name: "Video" }],
      ]),
      errors: [],
    });

    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n2:graph:Speaker"));

    const props = getCanvasProps();
    const secondNode = props.nodes.find((node) => node.id === "n2") as {
      position: { x: number; y: number };
      data: { resolvedTypes?: Record<string, string> };
    };
    expect(secondNode.position).toEqual({ x: 0, y: 0 });
    expect(secondNode.data.resolvedTypes).toEqual({
      "in.left": "Audio",
      "out.right": "Video",
    });

    fireEvent.click(screen.getByText("edit-first"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-edit-n1-"));
    fireEvent.click(screen.getByText("cancel-configuring"));

    await act(async () => {
      getCanvasProps().onDrop?.({
        preventDefault: vi.fn(),
        clientX: 10,
        clientY: 10,
        dataTransfer: {
          getData: (type: string) =>
            type === "application/openneuro"
              ? JSON.stringify({
                kind: "component",
                type_: "Mic",
                tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
                init: {
                  ignored: null,
                  refConfig: {
                    $ref: "#/$defs/RefConfig",
                    $defs: {
                      RefConfig: {
                        type: "object",
                        properties: {
                          gain: { type: "number" },
                        },
                      },
                    },
                  },
                  unionConfig: {
                    anyOf: [
                      { type: "string" },
                      {
                        type: "object",
                        properties: {
                          nested: { type: "string" },
                        },
                      },
                    ],
                  },
                },
                inputs: {},
                outputs: { output: "Audio" },
                ui_inputs: {},
                ui_outputs: {},
              })
              : "",
        },
      } as unknown as React.DragEvent);
    });

    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-"));
  });

  it("supports editing existing nodes and keyboard shortcuts in the group menu", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic"));

    fireEvent.click(screen.getByText("edit-first"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-edit-n1-"));
    fireEvent.click(screen.getByText("cancel-configuring"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).not.toHaveTextContent("configuring-edit-n1-"));

    fireEvent.click(screen.getByText("edit-first"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-edit-n1-"));
    fireEvent.click(screen.getByText("confirm-configuring"));
    await waitFor(() => expect(api.updateNodeInitArgs).toHaveBeenCalledWith("n1", { gain: 5 }));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).not.toHaveTextContent("configuring-edit-n1-"));

    fireEvent.click(screen.getByText("select-all"));
    fireEvent.click(screen.getByText("context-first"));
    fireEvent.click(screen.getByText("Group into subgraph"));
    fireEvent.change(screen.getByDisplayValue("Subgraph"), { target: { value: "KeyboardGroup" } });
    fireEvent.keyDown(screen.getByDisplayValue("KeyboardGroup"), { key: "Enter" });
    await waitFor(() => expect(api.createSubgraph).toHaveBeenCalledWith(expect.any(Array), "KeyboardGroup"));

    fireEvent.click(screen.getByText("select-all"));
    fireEvent.click(screen.getByText("context-first"));
    fireEvent.click(screen.getByText("Group into subgraph"));
    fireEvent.keyDown(screen.getByDisplayValue("Subgraph"), { key: "Escape" });
    await waitFor(() => expect(screen.queryByDisplayValue("Subgraph")).not.toBeInTheDocument());
  });

  it("opens configuration for anyOf object drops and removes the temp node on cancel", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic"));

    await act(async () => {
      getCanvasProps().onDrop?.({
        preventDefault: vi.fn(),
        clientX: 20,
        clientY: 30,
        dataTransfer: {
          getData: (type: string) =>
            type === "application/openneuro"
              ? JSON.stringify({
                kind: "component",
                type_: "Speaker",
                tags: { io: ["sink"], functionality: ["audio"], gpu: ["cpu"] },
                init: {
                  config: {
                    anyOf: [
                      {
                        type: "object",
                        properties: {
                          gain: { type: "number" },
                        },
                      },
                    ],
                  },
                },
                inputs: { input: "Audio" },
                outputs: {},
                ui_inputs: {},
                ui_outputs: {},
              })
              : "",
        },
      } as unknown as React.DragEvent);
    });

    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-"));
    fireEvent.click(screen.getByText("cancel-configuring"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).not.toHaveTextContent("configuring-"));
  });

  it("covers no-op graph events, text fallback drops, and default grouping names", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic"));

    fireEvent.click(screen.getByText("context-first"));
    expect(screen.getByText("Select 2+ nodes to group")).toBeInTheDocument();
    fireEvent.click(screen.getByText("pane-click"));

    const createEdgeSpy = vi.mocked(api.createEdge);
    createEdgeSpy.mockClear();
    act(() => {
      getCanvasProps().onConnect?.({
        source: "n1",
        target: "n2",
        sourceHandle: null,
        targetHandle: undefined,
      });
    });
    await waitFor(() => expect(api.createEdge).toHaveBeenCalledWith("n1", "", "n2", ""));

    createEdgeSpy.mockClear();
    act(() => {
      getCanvasProps().onConnect?.({
        source: "n1",
        sourceHandle: null,
        target: null,
        targetHandle: null,
      });
    });
    expect(api.createEdge).not.toHaveBeenCalled();

    const nodeSnapshot = screen.getByTestId("canvas-nodes").textContent;
    await act(async () => {
      getCanvasProps().onDrop?.({
        preventDefault: vi.fn(),
        clientX: 10,
        clientY: 10,
        dataTransfer: {
          getData: () => "",
        },
      } as unknown as React.DragEvent);
    });
    expect(screen.getByTestId("canvas-nodes").textContent).toBe(nodeSnapshot);

    await act(async () => {
      getCanvasProps().onDrop?.({
        preventDefault: vi.fn(),
        clientX: 10,
        clientY: 10,
        dataTransfer: {
          getData: (type: string) =>
            type === "text/plain"
              ? JSON.stringify({
                kind: "component",
                type_: "Speaker",
                tags: { io: ["sink"], functionality: ["audio"], gpu: ["cpu"] },
                init: {},
                inputs: { input: "Audio" },
                outputs: {},
                ui_inputs: {},
                ui_outputs: {},
              })
              : "",
        },
      } as unknown as React.DragEvent);
    });
    await waitFor(() => expect(api.createNode).toHaveBeenCalledWith("Speaker", undefined));

    await act(async () => {
      getCanvasProps().onDrop?.({
        preventDefault: vi.fn(),
        clientX: 10,
        clientY: 10,
        dataTransfer: {
          getData: (type: string) =>
            type === "application/openneuro" ? JSON.stringify({ kind: "unknown" }) : "",
        },
      } as unknown as React.DragEvent);
    });

    fireEvent.click(screen.getByText("drop-config"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-"));
    const configuringId = (getCanvasProps().nodes.find((node) => node.type === "configuring") as { id: string }).id;
    const deleteNodeSpy = vi.mocked(api.deleteNode);
    deleteNodeSpy.mockClear();
    act(() => {
      getCanvasProps().onNodesChange?.([{ type: "remove", id: configuringId }]);
    });
    expect(api.deleteNode).not.toHaveBeenCalled();

    const deleteEdgeSpy = vi.mocked(api.deleteEdge);
    deleteEdgeSpy.mockClear();
    act(() => {
      getCanvasProps().onEdgesChange?.([{ type: "remove", id: "missing-edge" }]);
    });
    expect(api.deleteEdge).not.toHaveBeenCalled();
    act(() => {
      getCanvasProps().onEdgesChange?.([{ type: "select", id: "missing-edge" }]);
    });

    fireEvent.click(screen.getByText("select-all"));
    fireEvent.click(screen.getByText("context-first"));
    fireEvent.click(screen.getByText("Group into subgraph"));
    fireEvent.change(screen.getByDisplayValue("Subgraph"), { target: { value: "   " } });
    fireEvent.click(screen.getByText("Create"));
    await waitFor(() => expect(api.createSubgraph).toHaveBeenCalledWith(expect.any(Array), "Subgraph"));
  });

  it("removes only edges touching a deleted node and accepts anyOf ref-backed config drops", async () => {
    vi.spyOn(api, "fetchEdges").mockResolvedValueOnce([
      {
        source_node: "n1",
        source_slot: "output",
        target_node: "n2",
        target_slot: "input",
      },
      {
        source_node: "n2",
        source_slot: "output",
        target_node: "n3",
        target_slot: "input",
      },
    ]);

    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-edges")).toHaveTextContent("n1:output->n2:input"));
    expect(screen.getByTestId("canvas-edges")).toHaveTextContent("n2:output->n3:input");

    fireEvent.click(screen.getByText("remove-node"));
    await waitFor(() => expect(api.deleteNode).toHaveBeenCalledWith("n1"));
    expect(screen.getByTestId("canvas-edges")).not.toHaveTextContent("n1:output->n2:input");
    expect(screen.getByTestId("canvas-edges")).toHaveTextContent("n2:output->n3:input");

    await act(async () => {
      getCanvasProps().onDrop?.({
        preventDefault: vi.fn(),
        clientX: 15,
        clientY: 25,
        dataTransfer: {
          getData: (type: string) =>
            type === "application/openneuro"
              ? JSON.stringify({
                kind: "component",
                type_: "Speaker",
                tags: { io: ["sink"], functionality: ["audio"], gpu: ["cpu"] },
                init: {
                  config: {
                    anyOf: [
                      { type: "string" },
                      {
                        $ref: "#/$defs/ConfigShape",
                      },
                    ],
                    $defs: {
                      ConfigShape: {
                        type: "object",
                        properties: {
                          gain: { type: "number" },
                        },
                      },
                    },
                  },
                },
                inputs: { input: "Audio" },
                outputs: {},
                ui_inputs: {},
                ui_outputs: {},
              })
              : "",
        },
      } as unknown as React.DragEvent);
    });

    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring-"));
  });

  it("stops the startup poll cleanly when the app unmounts before the backend responds", async () => {
    let resolveCurrentProject: ((value: { current_project: string | null }) => void) | undefined;
    vi.spyOn(api, "fetchCurrentProject").mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCurrentProject = resolve;
        }),
    );

    const { unmount } = render(<App />);
    unmount();

    await act(async () => {
      resolveCurrentProject?.({ current_project: "Alpha" });
      await Promise.resolve();
    });

    expect(api.startProject).not.toHaveBeenCalled();
  });

  it("logs initialization failures from the editor", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(api, "fetchNodes").mockRejectedValueOnce(new Error("init failed"));

    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(errorSpy).toHaveBeenCalledWith("[graph] Init failed:", expect.any(Error)));
  });

  it("logs group and ungroup failures from the editor", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    errorSpy.mockClear();
    vi.spyOn(api, "createSubgraph").mockRejectedValueOnce(new Error("group failed"));
    vi.spyOn(api, "ungroupNode").mockRejectedValueOnce(new Error("ungroup failed"));

    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic"));

    fireEvent.click(screen.getByText("drop-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("composite-1:graph:ProjectComposite"));
    fireEvent.click(screen.getByText("context-composite"));
    fireEvent.click(screen.getByText("Ungroup subgraph"));
    await waitFor(() => expect(errorSpy).toHaveBeenCalledWith("[graph] Ungroup failed:", expect.any(Error)));

    fireEvent.click(screen.getByText("select-all"));
    fireEvent.click(screen.getByText("context-first"));
    fireEvent.click(screen.getByText("Group into subgraph"));
    fireEvent.click(screen.getByText("Create"));
    await waitFor(() => expect(errorSpy).toHaveBeenCalledWith("[graph] Group failed:", expect.any(Error)));
  });

  it("fires the chooser cancel handler without leaving the loading state inconsistent", async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("cancel-project"));
    expect(screen.getByText("chooser:false")).toBeInTheDocument();
  });
});

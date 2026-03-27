import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App, { AppInner, deleteEdgeFromReactFlow, parseSlot, toReactFlowNode } from "./App";
import * as api from "@/lib/api";
import * as graphDataHook from "@/hooks/useGraphData";
import * as componentsHook from "@/hooks/useComponents";
import * as metricsHistoryHook from "@/hooks/useMetricsHistory";
import * as layout from "@/lib/layout";
import * as typecheck from "@/lib/typecheck";

let currentGraphCanvasProps: Record<string, unknown> | null = null;

vi.mock("@xyflow/react", async () => {
  const ReactModule = await import("react");
  return {
    ReactFlowProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    useNodesState: (initial: unknown[]) => {
      const [state, setState] = ReactModule.useState(initial);
      return [state, setState, vi.fn()];
    },
    useEdgesState: (initial: unknown[]) => {
      const [state, setState] = ReactModule.useState(initial);
      return [state, setState, vi.fn()];
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
    const nodes = (props.nodes as { id: string; type?: string; data?: Record<string, unknown> }[]) ?? [];
    const edges = (props.edges as { id: string }[]) ?? [];
    const firstGraphNode = nodes.find((node) => node.type === "graph");
    const compositeNode = nodes.find((node) => node.type === "graph" && node.data?.category === "composite");
    const configuringNode = nodes.find((node) => node.type === "configuring");
    const firstEdge = edges[0];

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
        <button onClick={() => firstGraphNode?.data?.onEditConfig?.()}>edit-first</button>
        <button onClick={() => configuringNode?.data?.onConfirm?.({ gain: 5 })}>confirm-configuring</button>
        <button onClick={() => configuringNode?.data?.onCancel?.()}>cancel-configuring</button>
        <button onClick={() => firstGraphNode && (props.onNodesChange as ((changes: unknown[]) => void) | undefined)?.([{ type: "remove", id: firstGraphNode.id }])}>
          remove-node
        </button>
        <button onClick={() => (props.onNodesChange as ((changes: unknown[]) => void) | undefined)?.([{ type: "remove", id: "configuring-temp" }])}>
          remove-configuring-node
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
            (props.onDragOver as ((event: React.DragEvent) => void) | undefined)?.({
              preventDefault: vi.fn(),
              stopPropagation: vi.fn(),
              dataTransfer: { dropEffect: "" },
            } as never)
          }
        >
          drag-over
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
          drop-component-simple
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
          drop-component-config
        </button>
        <button
          onClick={() =>
            (props.onDrop as ((event: React.DragEvent) => void) | undefined)?.(
              makeDropEvent("{invalid json") as never,
            )
          }
        >
          drop-invalid
        </button>
        <button
          onClick={() =>
            (props.onDrop as ((event: React.DragEvent) => void) | undefined)?.(
              makeDropEvent("", "") as never,
            )
          }
        >
          drop-empty
        </button>
        <button onClick={() => (props.onPaneClick as (() => void) | undefined)?.()}>pane-click</button>
      </div>
    );
  },
}));

vi.mock("@/components/graph/NodeSidebar", () => ({
  NodeSidebar: ({
    components,
    projects,
    currentProject,
  }: {
    components: unknown[];
    projects: unknown[];
    currentProject: string;
  }) => (
    <div>{`sidebar:${components.length}:${projects.length}:${currentProject}`}</div>
  ),
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
  ProjectComposite: {
    type_: "ProjectComposite",
    tags: { io: ["conduit"], functionality: ["other"], gpu: ["cpu"] },
    init: {},
    inputs: {},
    outputs: {},
    ui_inputs: {},
    ui_outputs: {},
  },
};

describe("App", () => {
  beforeEach(() => {
    currentGraphCanvasProps = null;
    vi.restoreAllMocks();

    vi.spyOn(componentsHook, "useComponents").mockReturnValue(components as never);
    vi.spyOn(graphDataHook, "useGraphData").mockReturnValue({
      connected: true,
      metrics: {
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
      },
      componentMap: componentMap as never,
    });
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
      errors: [
        {
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
          left: { kind: "concrete", name: "Audio" },
          right: { kind: "concrete", name: "Video" },
        },
      ],
    });

    vi.spyOn(api, "fetchProjects").mockResolvedValue([{ name: "Aux", has_thumbnail: false }]);
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
    vi.spyOn(api, "createNode").mockImplementation(async (type) => {
      if (type === "ProjectComposite") {
        return {
          id: "composite-1",
          type: "ProjectComposite",
          is_composite: true,
          status: "running",
          x: 0,
          y: 0,
          inputs: { "group.input": "Receiver[Audio]" },
          outputs: { "group.output": "Sender[Audio]" },
        };
      }
      return {
        id: `${type}-node`,
        type,
        is_composite: false,
        status: "startup",
        x: 0,
        y: 0,
        init_args: {},
      };
    });
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

  it("covers helper functions", async () => {
    expect(parseSlot(undefined)).toBe("");
    expect(parseSlot("out-audio-main")).toBe("audio-main");

    const deleteEdge = vi.spyOn(api, "deleteEdge").mockResolvedValue();
    deleteEdgeFromReactFlow({
      source: "a",
      sourceHandle: "out-src",
      target: "b",
      targetHandle: "in-dst",
    } as never);
    await act(async () => {});
    expect(deleteEdge).toHaveBeenCalledWith("a", "src", "b", "dst");

    expect(
      toReactFlowNode(
        {
          id: "c1",
          type: "ProjectComposite",
          is_composite: true,
          status: "running",
          x: 0,
          y: 0,
          inputs: { "c.in": "Receiver[Audio]" },
          outputs: { "c.out": "Sender[Audio]" },
        },
        { x: 1, y: 2 },
        componentMap as never,
        { inputs: {}, outputs: {} },
      ).data.category,
    ).toBe("composite");

    expect(
      toReactFlowNode(
        {
          id: "r1",
          type: "Mic",
          is_composite: false,
          status: "startup",
          x: 0,
          y: 0,
        },
        { x: 3, y: 4 },
        componentMap as never,
        { inputs: { Mic: {} }, outputs: { Mic: {} } },
      ).data.category,
    ).toBe("source");
  });

  it("renders chooser when there is no current project and opens the editor from it", async () => {
    vi.spyOn(api, "fetchCurrentProject").mockResolvedValue({ current_project: null });
    render(<App />);
    expect(screen.getByText("Connecting to backend...")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("open-project"));
    await waitFor(() =>
      expect(screen.getByText(/sidebar:2:\d+:OpenedProject/)).toBeInTheDocument(),
    );
  });

  it("retries when the backend is unavailable", async () => {
    vi.useFakeTimers();
    const fetchCurrentProject = vi.spyOn(api, "fetchCurrentProject");
    fetchCurrentProject
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ current_project: null });
    const timeoutSpy = vi.spyOn(globalThis, "setTimeout");

    render(<App />);
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchCurrentProject).toHaveBeenCalledTimes(1);
    expect(timeoutSpy).toHaveBeenCalledWith(expect.any(Function), 500);
  });

  it("falls back to chooser when auto-starting current project fails and supports chooser cancel", async () => {
    vi.spyOn(api, "fetchCurrentProject").mockResolvedValue({ current_project: "Alpha" });
    vi.spyOn(api, "startProject").mockRejectedValue(new Error("boom"));

    render(<App />);
    await waitFor(() => expect(screen.getByText("chooser:false")).toBeInTheDocument());
    fireEvent.click(screen.getByText("cancel-project"));
    expect(screen.getByText("chooser:false")).toBeInTheDocument();
  });

  it("initializes AppInner and handles editor interactions", async () => {
    const onGoHome = vi.fn();
    render(<AppInner projectName="Alpha" onGoHome={onGoHome} />);

    await waitFor(() => expect(screen.getByText("sidebar:2:1:Alpha")).toBeInTheDocument());
    expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic");
    expect(screen.getByTestId("canvas-edges")).toHaveTextContent("n1:output->n2:input");

    fireEvent.click(screen.getByText("toggle-logging"));
    expect(screen.getByText("logging:none")).toBeInTheDocument();

    fireEvent.click(screen.getByText("select-first"));
    expect(screen.getByText("logging:n1")).toBeInTheDocument();

    fireEvent.click(screen.getByText("open-dashboard"));
    expect(screen.getByText("close-dashboard")).toBeInTheDocument();
    fireEvent.click(screen.getByText("close-dashboard"));
    expect(screen.queryByText("close-dashboard")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("open-env"));
    expect(screen.getByText("close-env")).toBeInTheDocument();
    fireEvent.click(screen.getByText("close-env"));

    fireEvent.click(screen.getByTitle("Back to projects"));
    await waitFor(() => expect(api.closeProject).toHaveBeenCalled());
    expect(onGoHome).toHaveBeenCalled();

    fireEvent.click(screen.getByText("drag-over"));
    fireEvent.click(screen.getByText("connect"));
    fireEvent.click(screen.getByText("drag-stop"));
    fireEvent.click(screen.getByText("remove-edge"));
    fireEvent.click(screen.getByText("remove-configuring-node"));
    fireEvent.click(screen.getByText("remove-node"));

    await waitFor(() => expect(api.createEdge).toHaveBeenCalled());
    expect(api.updateNode).toHaveBeenCalledWith("n1", { x: 99, y: 101 });
    expect(api.deleteNode).toHaveBeenCalledWith("n1");
    expect(api.deleteEdge).toHaveBeenCalled();
  });

  it("handles drops, configuration flows, grouping, ungrouping, and pane dismissal", async () => {
    render(<AppInner projectName="Alpha" onGoHome={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("n1:graph:Mic"));

    fireEvent.click(screen.getByText("drop-empty"));
    fireEvent.click(screen.getByText("drop-invalid"));
    fireEvent.click(screen.getByText("drop-component-simple"));
    fireEvent.click(screen.getByText("drop-component-config"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring"));
    fireEvent.click(screen.getByText("confirm-configuring"));
    await waitFor(() => expect(api.createNode).toHaveBeenCalledWith("Mic", { gain: 5 }));

    fireEvent.click(screen.getByText("drop-component-config"));
    fireEvent.click(screen.getByText("cancel-configuring"));

    fireEvent.click(screen.getByText("edit-first"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("configuring"));
    fireEvent.click(screen.getByText("confirm-configuring"));
    await waitFor(() => expect(api.updateNodeInitArgs).toHaveBeenCalledWith("n1", { gain: 5 }));

    fireEvent.click(screen.getByText("drop-project"));
    await waitFor(() => expect(screen.getByTestId("canvas-nodes")).toHaveTextContent("composite-1:graph:ProjectComposite"));
    fireEvent.click(screen.getByText("context-composite"));
    fireEvent.click(screen.getByText("Ungroup subgraph"));
    await waitFor(() => expect(api.ungroupNode).toHaveBeenCalled());

    fireEvent.click(screen.getByText("context-first"));
    expect(screen.getByText("Select 2+ nodes to group")).toBeInTheDocument();
    fireEvent.click(screen.getByText("pane-click"));
    expect(screen.queryByText("Select 2+ nodes to group")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("select-all"));
    fireEvent.click(screen.getByText("context-first"));
    fireEvent.click(screen.getByText("Group into subgraph"));
    const nameInput = screen.getByDisplayValue("Subgraph");
    fireEvent.change(nameInput, { target: { value: "Grouped" } });
    fireEvent.keyDown(nameInput, { key: "Escape" });
    expect(screen.queryByText("Create")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText("context-first"));
    fireEvent.click(screen.getByText("Group into subgraph"));
    fireEvent.change(screen.getByDisplayValue("Subgraph"), { target: { value: "Grouped" } });
    fireEvent.click(screen.getByText("Create"));
    await waitFor(() => expect(api.createSubgraph).toHaveBeenCalledWith(expect.any(Array), "Grouped"));
  });
});

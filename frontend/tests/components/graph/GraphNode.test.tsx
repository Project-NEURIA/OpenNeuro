import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GraphNode } from "@/components/graph/GraphNode";
import * as textHook from "@/hooks/useUITextOutput";
import * as videoHook from "@/hooks/useUIVideoOutput";
import * as outputHook from "@/hooks/useUIOutput";
import * as inputHook from "@/hooks/useUIInput";
import * as channelContext from "@/contexts/UIChannelContext";

vi.mock("@xyflow/react", () => ({
  Handle: ({ id }: { id: string }) => <div data-testid={id} />,
  Position: {
    Left: "left",
    Right: "right",
  },
}));

describe("GraphNode", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders selected node widgets, handles, metrics, and sends UI input", () => {
    vi.spyOn(videoHook, "useUIVideoOutput").mockReturnValue("blob:frame");
    vi.spyOn(textHook, "useUITextOutput").mockReturnValue("transcript");
    vi.spyOn(outputHook, "useUIOutput").mockReturnValue({ status: "ok" });
    const sendGeneric = vi.fn();
    vi.spyOn(inputHook, "useUIInput").mockReturnValue(sendGeneric);
    const sendUIInput = vi.fn();
    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput,
      subscribe: vi.fn(() => vi.fn()),
    });
    const onEditConfig = vi.fn();

    render(
      <GraphNode
        id="node-1"
        selected
        dragging={false}
        zIndex={1}
        type="graph"
        isConnectable
        xPos={0}
        yPos={0}
        data={{
          label: "CompositeNode",
          category: "composite",
          onEditConfig,
          inputs: ["foo.audio"],
          outputs: ["bar.video"],
          inputTypes: { "foo.audio": { name: "InputType", optional: false } },
          outputTypes: { "bar.video": { name: "OutputType", optional: true } },
          resolvedTypes: {
            "in.foo.audio": "ResolvedInput",
            "out.bar.video": "ResolvedOutput",
          },
          status: "running",
          nodeMetrics: {
            name: "CompositeNode",
            status: "running",
            senders: {
              output: {
                name: "output",
                msg_count_delta: 4,
                byte_count_delta: 1024,
                last_send_time: 0,
                buffer_depth: 2,
              },
            },
            receivers: {
              input: {
                name: "input",
                msg_count_delta: 2,
                byte_count_delta: 256,
                lag: 1,
              },
            },
          },
          ui_inputs: {
            textIn: "UITextReceiver",
            genericIn: "UICustomReceiver",
          },
          ui_outputs: {
            videoOut: "UIVideoSender",
            textOut: "UITextSender",
            dataOut: "UICustomSender",
          },
        }}
      />,
    );

    expect(screen.getByText("CompositeNode")).toBeInTheDocument();
    fireEvent.click(screen.getByTitle("Edit settings"));
    expect(onEditConfig).toHaveBeenCalled();

    expect(screen.getByAltText("Video stream")).toHaveAttribute("src", "blob:frame");
    expect(screen.getByText("transcript")).toBeInTheDocument();
    expect(screen.getByText(/"status": "ok"/)).toBeInTheDocument();
    expect(screen.getByTestId("in-foo.audio")).toBeInTheDocument();
    expect(screen.getByTestId("out-bar.video")).toBeInTheDocument();
    expect(screen.getByText("audio")).toBeInTheDocument();
    expect(screen.getByText("video")).toBeInTheDocument();
    expect(screen.getByText("output")).toBeInTheDocument();
    expect(screen.getByText("input")).toBeInTheDocument();

    const textInput = screen.getByPlaceholderText("textIn...");
    fireEvent.change(textInput, { target: { value: "hello" } });
    fireEvent.keyDown(textInput, { key: "Enter" });
    expect(sendUIInput).toHaveBeenCalledWith("node-1", "textIn", "hello");

    const genericInput = screen.getByPlaceholderText("genericIn (JSON)...");
    fireEvent.change(genericInput, { target: { value: "{\"value\":1}" } });
    fireEvent.click(screen.getAllByRole("button", { name: "Send" })[1]!);
    fireEvent.change(genericInput, { target: { value: "raw" } });
    fireEvent.keyDown(genericInput, { key: "Enter" });
    expect(sendGeneric).toHaveBeenNthCalledWith(1, { value: 1 });
    expect(sendGeneric).toHaveBeenNthCalledWith(2, "raw");
  });

  it("renders fallback widgets and unknown categories without crashing", () => {
    vi.spyOn(videoHook, "useUIVideoOutput").mockReturnValue(null);
    vi.spyOn(textHook, "useUITextOutput").mockReturnValue(null);
    vi.spyOn(outputHook, "useUIOutput").mockReturnValue("raw text");
    vi.spyOn(inputHook, "useUIInput").mockReturnValue(vi.fn());
    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput: vi.fn(),
      subscribe: vi.fn(() => vi.fn()),
    });

    render(
      <GraphNode
        id="node-2"
        selected={false}
        dragging={false}
        zIndex={1}
        type="graph"
        isConnectable
        xPos={0}
        yPos={0}
        data={{
          label: "AlienNode",
          category: "alien",
          inputs: ["plain"],
          outputs: ["typed.out"],
          inputTypes: { plain: { name: "Input", optional: true } },
          outputTypes: { "typed.out": { name: "Output", optional: false } },
          status: "offline",
          nodeMetrics: { name: "AlienNode", status: "offline", senders: {}, receivers: {} },
          ui_outputs: {
            rawOut: "UICustomSender",
            videoOut: "UIVideoSender",
          },
        }}
      />,
    );

    expect(screen.getByText("alien")).toBeInTheDocument();
    expect(screen.getByText("raw text")).toBeInTheDocument();
    expect(screen.getByText("no signal")).toBeInTheDocument();
    expect(screen.getByText("out")).toBeInTheDocument();
  });

  it("shows the awaiting-data state and ignores empty submissions", () => {
    const sendGeneric = vi.fn();
    const sendUIInput = vi.fn();
    vi.spyOn(videoHook, "useUIVideoOutput").mockReturnValue(null);
    vi.spyOn(textHook, "useUITextOutput").mockReturnValue(null);
    vi.spyOn(outputHook, "useUIOutput").mockReturnValue(null);
    vi.spyOn(inputHook, "useUIInput").mockReturnValue(sendGeneric);
    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput,
      subscribe: vi.fn(() => vi.fn()),
    });

    render(
      <GraphNode
        id="node-3"
        selected={false}
        dragging={false}
        zIndex={1}
        type="graph"
        isConnectable
        xPos={0}
        yPos={0}
        data={{
          label: "BareNode",
          category: "source",
          inputs: [],
          outputs: [],
          inputTypes: {},
          outputTypes: {},
          status: "stopped",
          nodeMetrics: null,
          ui_inputs: {
            textIn: "UIKeystrokeReceiver",
            genericIn: "UICustomReceiver",
          },
        }}
      />,
    );

    expect(screen.getByText("awaiting data")).toBeInTheDocument();

    const textInput = screen.getByPlaceholderText("textIn...");
    fireEvent.change(textInput, { target: { value: "   " } });
    fireEvent.click(screen.getAllByRole("button", { name: "Send" })[0]!);

    const genericInput = screen.getByPlaceholderText("genericIn (JSON)...");
    fireEvent.change(genericInput, { target: { value: "   " } });
    fireEvent.click(screen.getAllByRole("button", { name: "Send" })[1]!);

    expect(sendUIInput).not.toHaveBeenCalled();
    expect(sendGeneric).not.toHaveBeenCalled();
  });

  it("covers sink styling and omits the widget section when no ui channels are present", () => {
    vi.spyOn(videoHook, "useUIVideoOutput").mockReturnValue(null);
    vi.spyOn(textHook, "useUITextOutput").mockReturnValue(null);
    vi.spyOn(outputHook, "useUIOutput").mockReturnValue(null);
    vi.spyOn(inputHook, "useUIInput").mockReturnValue(vi.fn());
    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput: vi.fn(),
      subscribe: vi.fn(() => vi.fn()),
    });

    render(
      <GraphNode
        id="node-4"
        selected={false}
        dragging={false}
        zIndex={1}
        type="graph"
        isConnectable
        xPos={0}
        yPos={0}
        data={{
          label: "SinkNode",
          category: "sink",
          inputs: ["input"],
          outputs: [],
          inputTypes: { input: { name: "Audio", optional: false } },
          outputTypes: {},
          status: "idle",
          nodeMetrics: {
            name: "SinkNode",
            status: "idle",
            senders: {},
            receivers: {},
          },
          ui_inputs: {},
          ui_outputs: {},
        }}
      />,
    );

    expect(screen.getByText("sink")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText(/JSON/)).not.toBeInTheDocument();
  });

  it("keeps non-enter keys and empty output payloads on the non-submitting path", () => {
    const sendGeneric = vi.fn();
    const sendUIInput = vi.fn();
    vi.spyOn(videoHook, "useUIVideoOutput").mockReturnValue(null);
    vi.spyOn(textHook, "useUITextOutput").mockReturnValue(null);
    vi.spyOn(outputHook, "useUIOutput").mockReturnValue(null);
    vi.spyOn(inputHook, "useUIInput").mockReturnValue(sendGeneric);
    vi.spyOn(channelContext, "useUIChannel").mockReturnValue({
      connected: true,
      sendUIInput,
      subscribe: vi.fn(() => vi.fn()),
    });

    render(
      <GraphNode
        id="node-5"
        selected={false}
        dragging={false}
        zIndex={1}
        type="graph"
        isConnectable
        xPos={0}
        yPos={0}
        data={{
          label: "QuietNode",
          category: "conduit",
          inputs: [],
          outputs: [],
          inputTypes: {},
          outputTypes: {},
          status: "stopped",
          nodeMetrics: null,
          ui_inputs: {
            textIn: "UITextReceiver",
            genericIn: "UICustomReceiver",
          },
          ui_outputs: {
            textOut: "UITextSender",
            dataOut: "UICustomSender",
          },
        }}
      />,
    );

    expect(screen.getAllByText("", { selector: "span, pre" }).length).toBeGreaterThan(0);

    const textInput = screen.getByPlaceholderText("textIn...");
    fireEvent.change(textInput, { target: { value: "hello" } });
    fireEvent.keyDown(textInput, { key: "Tab" });
    expect(sendUIInput).not.toHaveBeenCalled();

    const genericInput = screen.getByPlaceholderText("genericIn (JSON)...");
    fireEvent.change(genericInput, { target: { value: "{\"value\":2}" } });
    fireEvent.keyDown(genericInput, { key: "Escape" });
    expect(sendGeneric).not.toHaveBeenCalled();
  });
});

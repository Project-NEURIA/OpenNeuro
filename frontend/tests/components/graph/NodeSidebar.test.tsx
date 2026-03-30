import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NodeSidebar } from "@/components/graph/NodeSidebar";
import type { ComponentInfo } from "@/lib/types";

const components: ComponentInfo[] = [
  {
    type_: "Mic",
    description: "**bold** *italic* `code` tail",
    tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu", "nvidia"] },
    init: { config: {} },
    inputs: { input: "Audio" },
    outputs: { output: "Audio" },
    ui_inputs: {},
    ui_outputs: {},
  },
  {
    type_: "Speaker",
    tags: { io: ["sink"], functionality: ["audio"], gpu: ["cpu"] },
    init: {},
    inputs: { input: "Audio" },
    outputs: {},
    ui_inputs: {},
    ui_outputs: {},
  },
  {
    type_: "Current",
    tags: { io: ["conduit"], functionality: ["misc"], gpu: ["cpu"] },
    init: {},
    inputs: {},
    outputs: {},
    ui_inputs: {},
    ui_outputs: {},
    is_composite: true,
  },
  {
    type_: "Another",
    description: "plain hover text",
    tags: { io: ["conduit"], functionality: ["misc"], gpu: ["amd"] },
    init: {},
    inputs: {},
    outputs: {},
    ui_inputs: {},
    ui_outputs: {},
    is_composite: true,
  },
];

describe("NodeSidebar", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useFakeTimers();
  });

  it("searches, collapses, shows hover info, and serializes drag payloads", () => {
    const { container } = render(
      <NodeSidebar
        components={components}
        currentProject="Current"
      />,
    );

    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText("Sinks")).toBeInTheDocument();
    expect(screen.getByText("Projects")).toBeInTheDocument();
    expect(screen.queryByText("Current")).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Search..."), {
      target: { value: "mic" },
    });
    expect(screen.getByText("Mic")).toBeInTheDocument();
    expect(screen.queryByText("Speaker")).not.toBeInTheDocument();
    fireEvent.click(container.querySelector("button.absolute.right-2")!);
    expect((screen.getByPlaceholderText("Search...") as HTMLInputElement).value).toBe("");

    const item = screen.getByText("Mic").closest("div[draggable='true']")!;
    fireEvent.mouseEnter(item);
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(screen.getByText("bold")).toBeInTheDocument();
    expect(screen.getByText("italic")).toBeInTheDocument();
    expect(screen.getByText("code")).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes("tail"))).toBeInTheDocument();
    expect(screen.getByText("Inputs")).toBeInTheDocument();
    expect(screen.getByText("Outputs")).toBeInTheDocument();
    expect(screen.getByText("Config")).toBeInTheDocument();

    const dataTransfer = {
      setData: vi.fn(),
      effectAllowed: "",
    };
    fireEvent.dragStart(item, { dataTransfer });
    expect(dataTransfer.setData).toHaveBeenCalledWith(
      "application/openneuro",
      expect.stringContaining("\"kind\":\"component\""),
    );

    const projectItem = screen.getByText("Another").closest("div[draggable='true']")!;
    fireEvent.dragStart(projectItem, { dataTransfer });
    expect(dataTransfer.setData).toHaveBeenCalledWith(
      "application/openneuro",
      JSON.stringify({ kind: "project", name: "Another" }),
    );

    fireEvent.click(screen.getByText("Sources"));
    expect(screen.queryByText("Mic")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Sources"));

    fireEvent.click(screen.getAllByText("Audio")[0]!);
    expect(screen.queryByText("Mic")).not.toBeInTheDocument();
    fireEvent.click(screen.getAllByText("Audio")[0]!);

    fireEvent.click(screen.getByText("Projects"));
    expect(screen.queryByText("Another")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Projects"));
    expect(screen.getByText("Another")).toBeInTheDocument();
  });

  it("hides the info panel when the sidebar bounds are unavailable", () => {
    const mysteryComponents: ComponentInfo[] = [
      {
        type_: "MysteryWidget",
        description: "plain hover text",
        tags: { io: ["source"], functionality: ["misc"], gpu: ["intel"] },
        init: {},
        inputs: {},
        outputs: {},
        ui_inputs: {},
        ui_outputs: {},
      },
    ];

    const { container } = render(
      <NodeSidebar
        components={mysteryComponents}
        currentProject="Current"
      />,
    );

    const sidebar = container.querySelector("div.absolute.top-4.left-4.z-10.w-52") as HTMLDivElement;
    const item = screen.getByText("MysteryWidget").closest("div[draggable='true']")!;
    const rectSpy = vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function () {
      if (this === sidebar) {
        return undefined as never;
      }
      return {
        x: 0,
        y: 24,
        width: 180,
        height: 20,
        top: 24,
        right: 180,
        bottom: 44,
        left: 0,
        toJSON: () => ({}),
      } as DOMRect;
    });

    fireEvent.mouseEnter(item);
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(screen.queryByText("plain hover text")).not.toBeInTheDocument();

    rectSpy.mockRestore();
  });

  it("keeps hover cards alive while the pointer is over them and clears them after leaving", () => {
    const { container } = render(
      <NodeSidebar
        components={components}
        currentProject="Current"
      />,
    );

    const micItem = screen.getByText("Mic").closest("div[draggable='true']")!;
    fireEvent.mouseEnter(micItem);
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(screen.getByText("bold")).toBeInTheDocument();

    fireEvent.mouseLeave(micItem);
    const hoverWrapper = container.querySelector("div.fixed.z-50.w-64")?.parentElement as HTMLElement;
    fireEvent.mouseEnter(hoverWrapper);
    act(() => {
      vi.advanceTimersByTime(150);
    });
    expect(screen.getByText("bold")).toBeInTheDocument();

    fireEvent.mouseLeave(hoverWrapper);
    act(() => {
      vi.advanceTimersByTime(150);
    });
    expect(screen.queryByText("bold")).not.toBeInTheDocument();

    const projectItem = screen.getByText("Another").closest("div[draggable='true']")!;
    fireEvent.mouseEnter(projectItem);
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(screen.getByText("plain hover text")).toBeInTheDocument();
  });

  it("renders code-only markdown, unknown gpu tags, and multi-group components", () => {
    const mixedComponents: ComponentInfo[] = [
      {
        type_: "Hybrid",
        description: "`snippet`",
        tags: { io: ["source", "sink"], functionality: ["audio", "misc"], gpu: ["mystery" as never] },
        init: {},
        inputs: {},
        outputs: {},
        ui_inputs: {},
        ui_outputs: {},
      },
    ];

    render(
      <NodeSidebar
        components={mixedComponents}
        currentProject="Current"
      />,
    );

    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText("Sinks")).toBeInTheDocument();
    expect(screen.getAllByText("Audio").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Misc").length).toBeGreaterThan(0);

    const item = screen.getAllByText("Hybrid")[0]!.closest("div[draggable='true']")!;
    fireEvent.mouseEnter(item);
    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(screen.getByText("snippet")).toBeInTheDocument();
    expect(screen.getByText("mystery")).toBeInTheDocument();
  });
});

import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { __test__, NodeSidebar } from "@/components/graph/NodeSidebar";
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
];

describe("NodeSidebar", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.useFakeTimers();
  });

  it("searches, collapses, shows hover info, and serializes drag payloads", async () => {
    const { container } = render(
      <NodeSidebar
        components={components}
        projects={[
          { name: "Current", has_thumbnail: false },
          { name: "Another", has_thumbnail: false },
        ]}
        currentProject="Current"
      />,
    );

    expect(screen.getByText("Sources")).toBeInTheDocument();
    expect(screen.getByText("Sinks")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Search..."), {
      target: { value: "mic" },
    });
    expect(screen.getByText("Mic")).toBeInTheDocument();
    expect(screen.queryByText("Speaker")).not.toBeInTheDocument();
    fireEvent.click(container.querySelector("button.absolute.right-2")!);
    expect((screen.getByPlaceholderText("Search...") as HTMLInputElement).value).toBe("");

    fireEvent.change(screen.getByPlaceholderText("Search..."), {
      target: { value: "" },
    });

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
    const hoverWrapper = container.querySelector(".fixed.z-50.w-64")?.parentElement as HTMLElement;
    fireEvent.mouseEnter(hoverWrapper);
    fireEvent.mouseLeave(item);
    act(() => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.queryByText("Inputs")).not.toBeInTheDocument();

    fireEvent.mouseEnter(item);
    act(() => {
      vi.advanceTimersByTime(300);
    });

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

  it("covers helper fallbacks, unknown icons, and hover panels without sidebar bounds", () => {
    render(<__test__.InlineMarkdown text="plain description" />);
    expect(screen.getByText("plain description")).toBeInTheDocument();
    render(<__test__.InlineMarkdown text="`code` tail" />);
    expect(screen.getByText("code")).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes("tail"))).toBeInTheDocument();
    expect(__test__.parseInlineMarkdown("`code`")).toEqual([
      { type: "code", value: "code" },
    ]);
    expect(__test__.parseInlineMarkdown("**bold** *italic* `code` tail")).toEqual([
      { type: "bold", value: "bold" },
      { type: "text", value: " " },
      { type: "italic", value: "italic" },
      { type: "text", value: " " },
      { type: "code", value: "code" },
      { type: "text", value: " tail" },
    ]);

    expect(__test__.groupComponents([
      {
        type_: "MysteryA",
        tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
        init: {},
        inputs: {},
        outputs: {},
        ui_inputs: {},
        ui_outputs: {},
      },
      {
        type_: "MysteryB",
        tags: { io: ["source"], functionality: ["audio"], gpu: ["cpu"] },
        init: {},
        inputs: {},
        outputs: {},
        ui_inputs: {},
        ui_outputs: {},
      },
    ]).source.audio).toHaveLength(2);

    const mysteryComponents: ComponentInfo[] = [
      {
        type_: "MysteryWidget",
        description: "plain hover text",
        tags: { io: ["source"], functionality: ["misc"], gpu: ["quantum" as never] },
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
        projects={[{ name: "Current", has_thumbnail: false }]}
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
    fireEvent.mouseEnter(item);
    act(() => {
      vi.advanceTimersByTime(300);
    });

    expect(screen.getByText("plain hover text")).toBeInTheDocument();
    expect(screen.getByText("quantum")).toBeInTheDocument();
  });
});


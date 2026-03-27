import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NodeSidebar } from "./NodeSidebar";
import type { ComponentInfo } from "@/lib/types";

const components: ComponentInfo[] = [
  {
    type_: "Mic",
    description: "**bold** *italic* `code`",
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
    render(
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
  });
});

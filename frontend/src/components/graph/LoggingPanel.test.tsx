import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LoggingPanel } from "./LoggingPanel";
import * as logsHook from "@/hooks/useComponentLogs";

describe("LoggingPanel", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders empty state when no node is selected", () => {
    vi.spyOn(logsHook, "useComponentLogs").mockReturnValue({
      entries: [],
      error: null,
    });

    render(<LoggingPanel selectedNode={null} />);
    expect(screen.getByText("No component selected.")).toBeInTheDocument();
  });

  it("renders counts, entries, and errors for the selected node", () => {
    vi.spyOn(logsHook, "useComponentLogs").mockReturnValue({
      entries: [
        { seq: 1, timestamp: 1, stream: "stdout", text: "hello" },
        { seq: 2, timestamp: 2, stream: "stderr", text: "oops" },
      ],
      error: "failed",
    });

    render(
      <LoggingPanel
        selectedNode={{
          id: "node-1",
          data: { label: "Mic", status: "running" },
        } as never}
      />,
    );

    expect(screen.getByText("stdout 1 | stderr 1")).toBeInTheDocument();
    expect(screen.getByText("Mic")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("oops")).toBeInTheDocument();
    expect(screen.getByText(/failed to load logs/i)).toBeInTheDocument();
  });

  it("renders the empty selected-node log state", () => {
    vi.spyOn(logsHook, "useComponentLogs").mockReturnValue({
      entries: [],
      error: null,
    });

    render(
      <LoggingPanel
        selectedNode={{
          id: "node-2",
          data: { label: "Speaker", status: "stopped" },
        } as never}
      />,
    );

    expect(screen.getByText("Speaker")).toBeInTheDocument();
    expect(screen.getByText(/no logs captured yet/i)).toBeInTheDocument();
  });
});


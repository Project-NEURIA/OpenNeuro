import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MetricsOverlay } from "./MetricsOverlay";
import * as api from "@/lib/api";

describe("MetricsOverlay", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders disconnected state and supports optional callbacks", () => {
    render(<MetricsOverlay connected={false} metrics={null} />);
    expect(screen.getByText("Disconnected")).toBeInTheDocument();
    expect(screen.getByText("0/0")).toBeInTheDocument();
  });

  it("starts, stops, toggles logging, opens dashboard and env", async () => {
    const startAll = vi.spyOn(api, "startAll").mockResolvedValue();
    const stopAll = vi.spyOn(api, "stopAll").mockResolvedValue();
    const onToggleLogging = vi.fn();
    const onOpenDashboard = vi.fn();
    const onOpenEnv = vi.fn();

    const { rerender } = render(
      <MetricsOverlay
        connected
        metrics={{
          timestamp: 1,
          nodes: {
            a: { name: "A", status: "stopped", senders: {}, receivers: {} },
            b: { name: "B", status: "stopped", senders: {}, receivers: {} },
          },
        }}
        loggingOpen
        onToggleLogging={onToggleLogging}
        onOpenDashboard={onOpenDashboard}
        onOpenEnv={onOpenEnv}
      />,
    );

    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.getByText("0/2")).toBeInTheDocument();

    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]!);
    fireEvent.click(buttons[2]!);
    fireEvent.click(buttons[3]!);
    fireEvent.click(buttons[4]!);

    expect(startAll).toHaveBeenCalled();
    expect(onToggleLogging).toHaveBeenCalled();
    expect(onOpenDashboard).toHaveBeenCalled();
    expect(onOpenEnv).toHaveBeenCalled();

    rerender(
      <MetricsOverlay
        connected
        metrics={{
          timestamp: 1,
          nodes: {
            a: { name: "A", status: "running", senders: {}, receivers: {} },
          },
        }}
      />,
    );
    fireEvent.click(screen.getAllByRole("button")[1]!);
    expect(stopAll).toHaveBeenCalled();
  });

  it("renders the inactive logging button style when logging is closed", () => {
    render(
      <MetricsOverlay
        connected
        metrics={{
          timestamp: 1,
          nodes: {
            a: { name: "A", status: "running", senders: {}, receivers: {} },
          },
        }}
        loggingOpen={false}
        onToggleLogging={() => {}}
      />,
    );

    expect(screen.getByTitle("Open component logging")).toBeInTheDocument();
  });
});

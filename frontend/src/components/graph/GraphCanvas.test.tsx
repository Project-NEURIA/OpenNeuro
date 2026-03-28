import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { GraphCanvas } from "./GraphCanvas";

const reactFlowMock = vi.fn((props: Record<string, unknown>) => (
  <div>
    <div data-testid="selection">{String(props.selectionOnDrag)}</div>
    <div data-testid="pan-on-drag">{JSON.stringify(props.panOnDrag)}</div>
    <button onClick={() => (props.onViewportChange as (viewport: { x: number; y: number; zoom: number }) => void)?.({ x: 10.2, y: 5.6, zoom: 1.5 })}>
      viewport
    </button>
    <button onClick={() => (props.onDrop as (event: React.DragEvent) => void)?.({} as React.DragEvent)}>
      drop
    </button>
    <button onClick={() => (props.onDragOver as (event: React.DragEvent) => void)?.({} as React.DragEvent)}>
      dragover
    </button>
    <button onClick={() => (props.onPaneClick as () => void)?.()}>pane</button>
  </div>
));

vi.mock("@xyflow/react", () => ({
  ReactFlow: (props: Record<string, unknown>) => reactFlowMock(props),
  Background: () => <div data-testid="background" />,
  Controls: () => <div data-testid="controls" />,
}));

describe("GraphCanvas", () => {
  it("wires react flow props and updates selection mode with shift", () => {
    const onDrop = vi.fn();
    const onDragOver = vi.fn();
    const onPaneClick = vi.fn();
    const viewport = document.createElement("div");
    viewport.className = "react-flow__viewport";
    document.body.appendChild(viewport);

    render(
      <GraphCanvas
        nodes={[]}
        edges={[]}
        onNodesChange={vi.fn()}
        onEdgesChange={vi.fn()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onPaneClick={onPaneClick}
      />,
    );

    expect(screen.getByTestId("selection")).toHaveTextContent("false");
    expect(screen.getByTestId("pan-on-drag")).toHaveTextContent("[0]");

    fireEvent.keyDown(window, { key: "Shift" });
    expect(screen.getByTestId("selection")).toHaveTextContent("true");
    expect(screen.getByTestId("pan-on-drag")).toHaveTextContent("false");

    fireEvent.keyUp(window, { key: "Shift" });
    expect(screen.getByTestId("selection")).toHaveTextContent("false");

    fireEvent.keyDown(window, { key: "A" });
    fireEvent.keyUp(window, { key: "A" });
    expect(screen.getByTestId("selection")).toHaveTextContent("false");

    fireEvent.click(screen.getByText("viewport"));
    expect(viewport.style.transform).toBe("translate(10px, 6px) scale(1.5)");
    viewport.remove();
    fireEvent.click(screen.getByText("viewport"));

    fireEvent.click(screen.getByText("drop"));
    fireEvent.click(screen.getByText("dragover"));
    fireEvent.click(screen.getByText("pane"));
    expect(onDrop).toHaveBeenCalled();
    expect(onDragOver).toHaveBeenCalled();
    expect(onPaneClick).toHaveBeenCalled();
  });
});


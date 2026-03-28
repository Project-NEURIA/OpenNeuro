import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GraphEdge } from "./GraphEdge";

const baseEdgeMock = vi.fn(({ id, path, style }) => (
  <div data-testid={`edge-${id}`} data-path={path} data-stroke={style.stroke} data-width={style.strokeWidth} />
));

vi.mock("@xyflow/react", () => ({
  BaseEdge: (props: unknown) => baseEdgeMock(props),
  EdgeLabelRenderer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  getBezierPath: () => ["M0,0 L1,1", 10, 20],
}));

describe("GraphEdge", () => {
  beforeEach(() => {
    baseEdgeMock.mockClear();
  });

  it("renders throughput-based edge styling", () => {
    render(
      <GraphEdge
        id="edge-1"
        sourceX={0}
        sourceY={0}
        targetX={10}
        targetY={10}
        sourcePosition={"right" as never}
        targetPosition={"left" as never}
        data={{ byteDelta: 10_000 }}
      />,
    );

    expect(screen.getByTestId("edge-edge-1")).toHaveAttribute(
      "data-stroke",
      "hsl(0 80% 60%)",
    );
    expect(screen.getByTestId("edge-edge-1")).toHaveAttribute("data-width", "4");
  });

  it("renders type errors with label and error styling", () => {
    render(
      <GraphEdge
        id="edge-2"
        sourceX={0}
        sourceY={0}
        targetX={10}
        targetY={10}
        sourcePosition={"right" as never}
        targetPosition={"left" as never}
        data={{ byteDelta: 0, typeError: "A != B" }}
      />,
    );

    expect(screen.getByTestId("edge-edge-2")).toHaveAttribute(
      "data-stroke",
      "hsl(0 80% 50%)",
    );
    expect(screen.getByText("A != B")).toBeInTheDocument();
  });

  it("renders idle edges with default throughput styling when data is missing", () => {
    render(
      <GraphEdge
        id="edge-3"
        sourceX={0}
        sourceY={0}
        targetX={10}
        targetY={10}
        sourcePosition={"right" as never}
        targetPosition={"left" as never}
      />,
    );

    expect(screen.getByTestId("edge-edge-3")).toHaveAttribute(
      "data-stroke",
      "var(--edge)",
    );
  });
});


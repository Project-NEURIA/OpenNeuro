import { describe, expect, it, vi } from "vitest";
import { layoutNodes } from "./layout";

describe("layoutNodes", () => {
  it("lays out nodes in topological order", () => {
    const positions = layoutNodes(
      [{ id: "a" }, { id: "b" }, { id: "c" }],
      [
        { source: "a", target: "b" },
        { source: "b", target: "c" },
      ],
    );

    expect(positions.map((node) => node.id)).toEqual(["a", "b", "c"]);
    expect(positions[0]).toMatchObject({ x: 80 });
    expect(positions[1]?.x).toBe(360);
  });

  it("adds disconnected nodes after sorted nodes", () => {
    const positions = layoutNodes(
      [{ id: "start" }, { id: "loopA" }, { id: "loopB" }, { id: "solo" }],
      [
        { source: "loopA", target: "loopB" },
        { source: "loopB", target: "loopA" },
      ],
    );

    expect(positions.map((node) => node.id)).toEqual(["start", "solo", "loopA", "loopB"]);
  });

  it("handles missing edge sources and multi-parent targets", () => {
    const positions = layoutNodes(
      [{ id: "a" }, { id: "b" }, { id: "c" }],
      [
        { source: "missing", target: "a" },
        { source: "a", target: "c" },
        { source: "b", target: "c" },
      ],
    );

    expect(positions.map((node) => node.id)).toEqual(["b", "a", "c"]);
  });

  it("handles edges pointing to nodes outside the layout set", () => {
    const positions = layoutNodes(
      [{ id: "a" }],
      [{ source: "a", target: "ghost" }],
    );

    expect(positions[0]).toMatchObject({ id: "a", x: 80 });
  });

  it("uses the missing in-degree fallback when a queued target cannot be read back", () => {
    const realGet = Map.prototype.get;
    const getSpy = vi.spyOn(Map.prototype, "get").mockImplementation(function (key: string) {
      if (key === "ghost") {
        return undefined;
      }
      return realGet.call(this, key);
    });

    const positions = layoutNodes(
      [{ id: "a" }],
      [{ source: "a", target: "ghost" }],
    );

    getSpy.mockRestore();
    expect(positions[0]).toMatchObject({ id: "a", x: 80 });
  });
});

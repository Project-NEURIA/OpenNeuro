import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import {
  checkTypes,
  collectLeafNames,
  getConstraints,
  typeToString,
  warmSubtypeCache,
} from "./typecheck";
import type { Graph, SlotType } from "./types";

describe("typecheck", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("collects leaf type names from unions and constructors", () => {
    expect(collectLeafNames("VideoFrame")).toEqual(["VideoFrame"]);
    expect(collectLeafNames("Union[AudioFrame, ImageFrame]")).toEqual([
      "AudioFrame",
      "ImageFrame",
    ]);
    expect(collectLeafNames("AudioFrame | ImageFrame")).toEqual([
      "AudioFrame",
      "ImageFrame",
    ]);
  });

  it("warms the subtype cache and applies subtype relationships", async () => {
    vi.spyOn(api, "fetchIsSubtype").mockImplementation(async (sub, sup) => {
      return sub === "Dog" && sup === "Animal";
    });

    await warmSubtypeCache(["Dog", "Animal", "Cat"]);

    const graph: Graph = {
      nodes: {
        source: { type: "Producer", init_args: {}, x: 0, y: 0 },
        sink: { type: "Consumer", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "source",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
      ],
    };

    const outputs: Record<string, Record<string, SlotType>> = {
      Producer: { out: { name: "Dog", optional: false } },
    };
    const inputs: Record<string, Record<string, SlotType>> = {
      Consumer: { in: { name: "Animal", optional: false } },
    };

    const result = checkTypes(graph, inputs, outputs, new Set(["Dog", "Animal", "Cat"]));
    expect(result.errors).toHaveLength(0);
    expect(typeToString(result.types.get("source.out.out")!)).toBe("Dog");
    expect(typeToString(result.types.get("sink.in.in")!)).toBe("Animal");
  });

  it("builds constraints and reports mismatched constructors and unknown slots", () => {
    const graph: Graph = {
      nodes: {
        source: { type: "Producer", init_args: {}, x: 0, y: 0 },
        sink: { type: "Consumer", init_args: {}, x: 0, y: 0 },
        unknown: { type: "Unknown", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "source",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
        {
          source_node: "unknown",
          source_slot: "missing",
          target_node: "sink",
          target_slot: "missing",
        },
      ],
    };

    const outputs: Record<string, Record<string, SlotType>> = {
      Producer: { out: { name: "List[Frame]", optional: false } },
    };
    const inputs: Record<string, Record<string, SlotType>> = {
      Consumer: { in: { name: "Map[Frame]", optional: false } },
    };

    const constraints = getConstraints(graph, inputs, outputs, new Set(["Frame"]));
    expect(constraints).toHaveLength(2);

    const result = checkTypes(graph, inputs, outputs, new Set(["Frame"]));
    expect(result.errors).toHaveLength(1);
    expect(typeToString(result.errors[0]!.left)).toBe("List[Frame]");
    expect(typeToString(result.errors[0]!.right)).toBe("Map[Frame]");
  });

  it("resolves unions and variables from constraints", () => {
    const graph: Graph = {
      nodes: {
        sourceA: { type: "A", init_args: {}, x: 0, y: 0 },
        sourceB: { type: "B", init_args: {}, x: 0, y: 0 },
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "sourceA",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
        {
          source_node: "sourceB",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
      ],
    };

    const outputs: Record<string, Record<string, SlotType>> = {
      A: { out: { name: "Cat", optional: false } },
      B: { out: { name: "Dog", optional: false } },
    };
    const inputs: Record<string, Record<string, SlotType>> = {
      Sink: { in: { name: "T", optional: false } },
    };

    const result = checkTypes(graph, inputs, outputs, new Set(["Cat", "Dog"]));
    expect(result.errors).toHaveLength(0);
    expect(typeToString(result.types.get("sink.in.in")!)).toBe("Cat | Dog");
    expect(typeToString(result.types.get("sourceA.out.out")!)).toBe("Cat");
  });
});

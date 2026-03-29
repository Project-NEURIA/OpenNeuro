import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/lib/api";
import {
  checkTypes,
  collectLeafNames,
  getConstraints,
  typeToString,
  warmSubtypeCache,
} from "@/lib/typecheck";
import type { Graph, SlotType } from "@/lib/types";

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

  it("handles constructor and union subtype inference through the public solver", async () => {
    vi.spyOn(api, "fetchIsSubtype").mockImplementation(async (sub, sup) => {
      return (
        (sub === "Cat" && sup === "Animal") ||
        (sub === "Dog" && sup === "Animal") ||
        (sub === "ImageFrame" && sup === "Frame")
      );
    });

    await warmSubtypeCache(["Cat", "Dog", "Animal", "ImageFrame", "Frame"]);

    const graph: Graph = {
      nodes: {
        listSource: { type: "ListSource", init_args: {}, x: 0, y: 0 },
        imageSource: { type: "ImageSource", init_args: {}, x: 0, y: 0 },
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "listSource",
          source_slot: "out",
          target_node: "sink",
          target_slot: "listIn",
        },
        {
          source_node: "imageSource",
          source_slot: "out",
          target_node: "sink",
          target_slot: "frameIn",
        },
      ],
    };
    const inputs: Record<string, Record<string, SlotType>> = {
      Sink: {
        listIn: { name: "List[Animal]", optional: false },
        frameIn: { name: "Frame | Animal", optional: false },
      },
    };
    const outputs: Record<string, Record<string, SlotType>> = {
      ListSource: { out: { name: "List[Cat]", optional: false } },
      ImageSource: { out: { name: "ImageFrame", optional: false } },
    };

    const result = checkTypes(graph, inputs, outputs, new Set(["Cat", "Dog", "Animal", "ImageFrame", "Frame"]));
    expect(result.errors).toHaveLength(0);
    expect(typeToString(result.types.get("listSource.out.out")!)).toBe("List[Cat]");
    expect(typeToString(result.types.get("sink.in.listIn")!)).toBe("List[Animal]");
    expect(typeToString(result.types.get("sink.in.frameIn")!)).toBe("Frame | Animal");
  });

  it("keeps parser and constraint edge cases observable through public APIs", () => {
    expect(collectLeafNames("Map[Cat, Dog]")).toEqual(["Cat", "Dog"]);

    const graph: Graph = {
      nodes: {
        source: { type: "Source", init_args: {}, x: 0, y: 0 },
        sink: { type: "Consumer", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "source",
          source_slot: "out",
          target_node: "sink",
          target_slot: "input",
        },
      ],
    };
    const inputs: Record<string, Record<string, SlotType>> = {
      Consumer: { input: { name: "Blue", optional: false } },
    };
    const outputs: Record<string, Record<string, SlotType>> = {
      Source: { out: { name: "Red", optional: false } },
    };
    const result = checkTypes(graph, inputs, outputs, new Set(["Red", "Blue"]));
    expect(result.errors).toHaveLength(1);
    expect(typeToString(result.errors[0]!.left)).toBe("Red");
    expect(typeToString(result.errors[0]!.right)).toBe("Blue");
  });
});


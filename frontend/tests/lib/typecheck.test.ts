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

  it("treats equivalent unions as equal even when their members are reordered", () => {
    const graph: Graph = {
      nodes: {
        source: { type: "Source", init_args: {}, x: 0, y: 0 },
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
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

    const inputs: Record<string, Record<string, SlotType>> = {
      Sink: { in: { name: "Cat | Dog", optional: false } },
    };
    const outputs: Record<string, Record<string, SlotType>> = {
      Source: { out: { name: "Union[Dog, Cat]", optional: false } },
    };

    const result = checkTypes(graph, inputs, outputs, new Set(["Cat", "Dog"]));
    expect(result.errors).toHaveLength(0);
    expect(typeToString(result.types.get("source.out.out")!)).toBe("Dog | Cat");
    expect(typeToString(result.types.get("sink.in.in")!)).toBe("Cat | Dog");
  });

  it("propagates variable bounds in both directions and resolves upper-bound-only variables", async () => {
    vi.spyOn(api, "fetchIsSubtype").mockImplementation(async (sub, sup) => {
      return sub === "Cat" && sup === "Animal";
    });

    await warmSubtypeCache(["Cat", "Animal"]);

    const propagationGraph: Graph = {
      nodes: {
        source: { type: "Source", init_args: {}, x: 0, y: 0 },
        middle: { type: "Middle", init_args: {}, x: 0, y: 0 },
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "middle",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
        {
          source_node: "source",
          source_slot: "out",
          target_node: "middle",
          target_slot: "in",
        },
      ],
    };
    const propagationInputs: Record<string, Record<string, SlotType>> = {
      Middle: { in: { name: "T", optional: false } },
      Sink: { in: { name: "Animal", optional: false } },
    };
    const propagationOutputs: Record<string, Record<string, SlotType>> = {
      Source: { out: { name: "Cat", optional: false } },
      Middle: { out: { name: "T", optional: false } },
    };

    const propagationResult = checkTypes(
      propagationGraph,
      propagationInputs,
      propagationOutputs,
      new Set(["Cat", "Animal"]),
    );
    expect(propagationResult.errors).toHaveLength(0);
    expect(typeToString(propagationResult.types.get("middle.in.in")!)).toBe("Cat");
    expect(typeToString(propagationResult.types.get("middle.out.out")!)).toBe("Cat");

    const selfGraph: Graph = {
      nodes: {
        loop: { type: "Loop", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "loop",
          source_slot: "out",
          target_node: "loop",
          target_slot: "in",
        },
      ],
    };
    const selfSlots: Record<string, Record<string, SlotType>> = {
      Loop: {
        in: { name: "T", optional: false },
        out: { name: "T", optional: false },
      },
    };
    const selfResult = checkTypes(selfGraph, { Loop: { in: selfSlots.Loop!.in } }, { Loop: { out: selfSlots.Loop!.out } }, new Set());
    expect(selfResult.errors).toHaveLength(0);
    expect(typeToString(selfResult.types.get("loop.in.in")!)).toBe("loop.T");

    const upperOnlyGraph: Graph = {
      nodes: {
        source: { type: "UpperSource", init_args: {}, x: 0, y: 0 },
        sink: { type: "UpperSink", init_args: {}, x: 0, y: 0 },
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
    const upperOnlyResult = checkTypes(
      upperOnlyGraph,
      { UpperSink: { in: { name: "Animal", optional: false } } },
      { UpperSource: { out: { name: "T", optional: false } } },
      new Set(["Animal"]),
    );
    expect(upperOnlyResult.errors).toHaveLength(0);
    expect(typeToString(upperOnlyResult.types.get("source.out.out")!)).toBe("Animal");
  });

  it("checks constructor subtyping through unions and reports failing unions", async () => {
    vi.spyOn(api, "fetchIsSubtype").mockImplementation(async (sub, sup) => {
      return (sub === "Cat" && sup === "Animal") || (sub === "Dog" && sup === "Animal");
    });

    await warmSubtypeCache(["Cat", "Dog", "Animal", "Rock"]);

    const constructorGraph: Graph = {
      nodes: {
        listSource: { type: "ListSource", init_args: {}, x: 0, y: 0 },
        listSink: { type: "ListSink", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "listSource",
          source_slot: "out",
          target_node: "listSink",
          target_slot: "in",
        },
      ],
    };
    const constructorResult = checkTypes(
      constructorGraph,
      { ListSink: { in: { name: "List[Animal] | Rock", optional: false } } },
      { ListSource: { out: { name: "List[Cat]", optional: false } } },
      new Set(["Cat", "Animal", "Rock"]),
    );
    expect(constructorResult.errors).toHaveLength(0);

    const failingUnionGraph: Graph = {
      nodes: {
        source: { type: "UnionSource", init_args: {}, x: 0, y: 0 },
        sink: { type: "AnimalSink", init_args: {}, x: 0, y: 0 },
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
    const failingResult = checkTypes(
      failingUnionGraph,
      { AnimalSink: { in: { name: "Animal", optional: false } } },
      { UnionSource: { out: { name: "Cat | Rock", optional: false } } },
      new Set(["Cat", "Animal", "Rock"]),
    );
    expect(failingResult.errors).toHaveLength(1);
    expect(typeToString(failingResult.errors[0]!.left)).toBe("Cat | Rock");
    expect(typeToString(failingResult.errors[0]!.right)).toBe("Animal");
  });

  it("handles repeated constraints, missing nodes, and union mismatch branches", () => {
    const repeatedGraph: Graph = {
      nodes: {
        source: { type: "Source", init_args: {}, x: 0, y: 0 },
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "source",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
        {
          source_node: "source",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
      ],
    };

    const repeatedResult = checkTypes(
      repeatedGraph,
      { Sink: { in: { name: "Animal", optional: false } } },
      { Source: { out: { name: "Animal", optional: false } } },
      new Set(["Animal"]),
    );
    expect(repeatedResult.errors).toHaveLength(0);

    const missingNodeGraph: Graph = {
      nodes: {
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "ghost",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
      ],
    };
    const missingNodeResult = checkTypes(
      missingNodeGraph,
      { Sink: { in: { name: "Audio", optional: false } } },
      {},
      new Set(["Audio"]),
    );
    expect(typeToString(missingNodeResult.errors[0]!.left)).toBe("?");

    const mismatchedUnionResult = checkTypes(
      {
        nodes: {
          source: { type: "UnionSource", init_args: {}, x: 0, y: 0 },
          sink: { type: "UnionSink", init_args: {}, x: 0, y: 0 },
        },
        edges: [
          {
            source_node: "source",
            source_slot: "out",
            target_node: "sink",
            target_slot: "in",
          },
        ],
      },
      { UnionSink: { in: { name: "Cat", optional: false } } },
      { UnionSource: { out: { name: "Cat | Dog", optional: false } } },
      new Set(["Cat", "Dog"]),
    );
    expect(mismatchedUnionResult.errors).toHaveLength(1);
    expect(typeToString(mismatchedUnionResult.errors[0]!.left)).toBe("Cat | Dog");
  });

  it("coalesces duplicate lower bounds and preserves unconstrained variables", () => {
    const duplicateGraph: Graph = {
      nodes: {
        sourceA: { type: "A", init_args: {}, x: 0, y: 0 },
        sourceB: { type: "B", init_args: {}, x: 0, y: 0 },
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
        free: { type: "Free", init_args: {}, x: 0, y: 0 },
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

    const duplicateResult = checkTypes(
      duplicateGraph,
      { Sink: { in: { name: "T", optional: false } } },
      {
        A: { out: { name: "Cat | Cat", optional: false } },
        B: { out: { name: "Cat", optional: false } },
        Free: { out: { name: "Loose", optional: false } },
      },
      new Set(["Cat"]),
    );

    expect(duplicateResult.errors).toHaveLength(0);
    expect(typeToString(duplicateResult.types.get("sink.in.in")!)).toBe("Cat");
    expect(typeToString(duplicateResult.types.get("free.out.out")!)).toBe("free.Loose");
  });

  it("propagates lower bounds before upper bounds and reports non-subtype fallthroughs", async () => {
    vi.spyOn(api, "fetchIsSubtype").mockImplementation(async (sub, sup) => {
      return sub === "Cat" && sup === "Animal";
    });

    await warmSubtypeCache(["Cat", "Animal"]);

    const lowerFirstGraph: Graph = {
      nodes: {
        source: { type: "Source", init_args: {}, x: 0, y: 0 },
        middle: { type: "Middle", init_args: {}, x: 0, y: 0 },
        sink: { type: "Sink", init_args: {}, x: 0, y: 0 },
      },
      edges: [
        {
          source_node: "source",
          source_slot: "out",
          target_node: "middle",
          target_slot: "in",
        },
        {
          source_node: "middle",
          source_slot: "out",
          target_node: "sink",
          target_slot: "in",
        },
      ],
    };

    const lowerFirstResult = checkTypes(
      lowerFirstGraph,
      {
        Middle: { in: { name: "T", optional: false } },
        Sink: { in: { name: "Animal", optional: false } },
      },
      {
        Source: { out: { name: "Cat", optional: false } },
        Middle: { out: { name: "T", optional: false } },
      },
      new Set(["Cat", "Animal"]),
    );
    expect(lowerFirstResult.errors).toHaveLength(0);
    expect(typeToString(lowerFirstResult.types.get("middle.out.out")!)).toBe("Cat");

    const fallthroughResult = checkTypes(
      {
        nodes: {
          source: { type: "BadSource", init_args: {}, x: 0, y: 0 },
          sink: { type: "BadSink", init_args: {}, x: 0, y: 0 },
        },
        edges: [
          {
            source_node: "source",
            source_slot: "out",
            target_node: "sink",
            target_slot: "in",
          },
        ],
      },
      { BadSink: { in: { name: "Animal", optional: false } } },
      { BadSource: { out: { name: "List[Cat]", optional: false } } },
      new Set(["Cat", "Animal"]),
    );
    expect(fallthroughResult.errors).toHaveLength(1);
    expect(typeToString(fallthroughResult.errors[0]!.left)).toBe("List[Cat]");
  });
});


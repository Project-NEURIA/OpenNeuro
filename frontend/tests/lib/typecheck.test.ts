import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/lib/api";
import {
  __test__,
  checkTypes,
  collectLeafNames,
  getConstraints,
  typeToString,
  warmSubtypeCache,
} from "@/lib/typecheck";
import type { Type } from "@/lib/typecheck";
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

  it("covers internal lattice and parser helpers", async () => {
    vi.spyOn(api, "fetchIsSubtype").mockImplementation(async (sub, sup) => {
      return (
        (sub === "Cat" && sup === "Animal") ||
        (sub === "Dog" && sup === "Animal") ||
        (sub === "ImageFrame" && sup === "Frame")
      );
    });

    await warmSubtypeCache(["Cat", "Dog", "Animal", "ImageFrame", "Frame"]);

    expect(__test__.flattenUnion([
      { kind: "concrete", name: "Cat" },
      { kind: "union", types: [{ kind: "concrete", name: "Dog" }] },
    ])).toEqual([
      { kind: "concrete", name: "Cat" },
      { kind: "concrete", name: "Dog" },
    ]);

    expect(__test__.typesEqual(
      { kind: "var", name: "T" },
      { kind: "var", name: "T" },
    )).toBe(true);
    expect(__test__.typesEqual(
      { kind: "constructor", name: "List", inner: { kind: "concrete", name: "Cat" } },
      { kind: "constructor", name: "List", inner: { kind: "concrete", name: "Cat" } },
    )).toBe(true);
    expect(__test__.typesEqual(
      {
        kind: "union",
        types: [
          { kind: "concrete", name: "Cat" },
          { kind: "concrete", name: "Dog" },
        ],
      },
      {
        kind: "union",
        types: [
          { kind: "concrete", name: "Dog" },
          { kind: "concrete", name: "Cat" },
        ],
      },
    )).toBe(true);
    expect(__test__.dedup([
      { kind: "concrete", name: "Cat" },
      { kind: "concrete", name: "Cat" },
    ])).toEqual([{ kind: "concrete", name: "Cat" }]);

    expect(__test__.isSubtype(
      { kind: "constructor", name: "Box", inner: { kind: "concrete", name: "Cat" } },
      { kind: "constructor", name: "Box", inner: { kind: "concrete", name: "Animal" } },
    )).toBe(true);
    expect(__test__.isSubtype(
      { kind: "concrete", name: "Cat" },
      {
        kind: "union",
        types: [
          { kind: "concrete", name: "Dog" },
          { kind: "concrete", name: "Animal" },
        ],
      },
    )).toBe(true);
    expect(__test__.isSubtype(
      {
        kind: "union",
        types: [
          { kind: "concrete", name: "Cat" },
          { kind: "concrete", name: "Dog" },
        ],
      },
      { kind: "concrete", name: "Animal" },
    )).toBe(true);
    expect(__test__.isSubtype(
      { kind: "var", name: "T" },
      { kind: "concrete", name: "Animal" },
    )).toBe(false);
    expect(__test__.join(
      { kind: "concrete", name: "Cat" },
      { kind: "concrete", name: "Animal" },
    )).toEqual({ kind: "concrete", name: "Animal" });
    expect(__test__.join(
      { kind: "concrete", name: "Animal" },
      { kind: "concrete", name: "Cat" },
    )).toEqual({ kind: "concrete", name: "Animal" });

    expect(__test__.parseType("Cat | Dog", new Set(["Cat", "Dog"]))).toEqual({
      kind: "union",
      types: [
        { kind: "concrete", name: "Cat" },
        { kind: "concrete", name: "Dog" },
      ],
    });
    expect(__test__.parseType("Union[Cat, Dog]", new Set(["Cat", "Dog"]))).toEqual({
      kind: "union",
      types: [
        { kind: "concrete", name: "Cat" },
        { kind: "concrete", name: "Dog" },
      ],
    });
    expect(__test__.parseType("List[Cat]", new Set(["Cat"]))).toEqual({
      kind: "constructor",
      name: "List",
      inner: { kind: "concrete", name: "Cat" },
    });
    expect(__test__.parseType("T", new Set(), "nodeA")).toEqual({
      kind: "var",
      name: "nodeA.T",
    });

    const graph: Graph = {
      nodes: {
        nodeA: { type: "Producer", init_args: {}, x: 0, y: 0 },
      },
      edges: [],
    };
    const outputs: Record<string, Record<string, SlotType>> = {
      Producer: { out: { name: "ImageFrame", optional: false } },
    };
    expect(__test__.slotType(graph, "missing", "out", "out", {}, outputs, new Set(["ImageFrame"]))).toEqual({
      kind: "concrete",
      name: "?",
    });
    expect(__test__.slotType(graph, "nodeA", "out", "missing", {}, outputs, new Set(["ImageFrame"]))).toEqual({
      kind: "concrete",
      name: "?",
    });
    expect(__test__.slotType(graph, "nodeA", "out", "out", {}, outputs, new Set(["ImageFrame"]))).toEqual({
      kind: "concrete",
      name: "ImageFrame",
    });

    const bounds = new Map<string, { lower: Type[]; upper: Type[] }>();
    const leftVarBounds = __test__.getBounds(bounds as never, "L");
    leftVarBounds.lower.push({ kind: "concrete", name: "Cat" });
    const errors: { left: Type; right: Type }[] = [];
    __test__.constrain(
      { kind: "var", name: "L" },
      { kind: "concrete", name: "Animal" },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      bounds as never,
      errors as never,
      new Set(),
    );
    expect(__test__.getBounds(bounds as never, "L").upper).toContainEqual({
      kind: "concrete",
      name: "Animal",
    });

    const rightVarBounds = __test__.getBounds(bounds as never, "R");
    rightVarBounds.upper.push({ kind: "concrete", name: "Animal" });
    __test__.constrain(
      { kind: "concrete", name: "Cat" },
      { kind: "var", name: "R" },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      bounds as never,
      errors as never,
      new Set(),
    );
    expect(__test__.getBounds(bounds as never, "R").lower).toContainEqual({
      kind: "concrete",
      name: "Cat",
    });

    __test__.constrain(
      { kind: "constructor", name: "List", inner: { kind: "concrete", name: "Cat" } },
      { kind: "constructor", name: "List", inner: { kind: "concrete", name: "Animal" } },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      bounds as never,
      errors as never,
      new Set(),
    );
    __test__.constrain(
      { kind: "constructor", name: "List", inner: { kind: "concrete", name: "Cat" } },
      { kind: "constructor", name: "Map", inner: { kind: "concrete", name: "Cat" } },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      bounds as never,
      errors as never,
      new Set(),
    );
    __test__.constrain(
      { kind: "concrete", name: "Dog" },
      { kind: "concrete", name: "ImageFrame" },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      bounds as never,
      errors as never,
      new Set(),
    );
    expect(errors).toHaveLength(2);

    expect(__test__.typeKey({
      kind: "union",
      types: [
        { kind: "concrete", name: "Dog" },
        { kind: "concrete", name: "Cat" },
      ],
    })).toBe("u:(c:Cat|c:Dog)");

    expect(__test__.applySubst(
      new Map([
        ["T", { kind: "constructor", name: "List", inner: { kind: "var", name: "U" } }],
        ["U", { kind: "concrete", name: "Cat" }],
      ]),
      { kind: "var", name: "T" },
    )).toEqual({
      kind: "constructor",
      name: "List",
      inner: { kind: "concrete", name: "Cat" },
    });

    expect(__test__.coalesce(new Map([
      ["Lower", { lower: [{ kind: "concrete", name: "Cat" }, { kind: "concrete", name: "Dog" }], upper: [] }],
      ["Upper", { lower: [], upper: [{ kind: "var", name: "T" }, { kind: "concrete", name: "Animal" }] }],
    ]))).toEqual(new Map([
      ["Lower", {
        kind: "union",
        types: [
          { kind: "concrete", name: "Cat" },
          { kind: "concrete", name: "Dog" },
        ],
      }],
      ["Upper", { kind: "concrete", name: "Animal" }],
    ]));
  });

  it("covers remaining parser and substitution edge cases", () => {
    expect(__test__.typesEqual(
      { kind: "union", types: [{ kind: "concrete", name: "Cat" }] },
      {
        kind: "union",
        types: [
          { kind: "concrete", name: "Cat" },
          { kind: "concrete", name: "Dog" },
        ],
      },
    )).toBe(false);

    expect(__test__.splitTopLevel("Box[List[Cat], Map[Dog]]")).toEqual([
      "Box[List[Cat], Map[Dog]]",
    ]);

    expect(__test__.parseType("Union[Cat]", new Set(["Cat"]))).toEqual({
      kind: "concrete",
      name: "Cat",
    });
    expect(__test__.parseType("Map[Cat, Dog]", new Set(["Cat", "Dog"]))).toEqual({
      kind: "var",
      name: "Map[Cat, Dog]",
    });
    expect(__test__.parseType("T", new Set())).toEqual({
      kind: "var",
      name: "T",
    });

    const graph: Graph = {
      nodes: {
        sink: { type: "Consumer", init_args: {}, x: 0, y: 0 },
      },
      edges: [],
    };
    expect(__test__.slotType(
      graph,
      "sink",
      "in",
      "input",
      { Consumer: { input: { name: "Cat", optional: false } } },
      {},
      new Set(["Cat"]),
    )).toEqual({
      kind: "concrete",
      name: "Cat",
    });

    const cache = new Set<string>();
    const bounds = new Map<string, { lower: Type[]; upper: Type[] }>();
    const errors: Type[] = [];
    __test__.constrain(
      { kind: "concrete", name: "Red" },
      { kind: "concrete", name: "Blue" },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      bounds as never,
      errors as never,
      cache,
    );
    __test__.constrain(
      { kind: "concrete", name: "Red" },
      { kind: "concrete", name: "Blue" },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      bounds as never,
      errors as never,
      cache,
    );
    expect(errors).toHaveLength(1);

    const sameVarBounds = new Map<string, { lower: Type[]; upper: Type[] }>();
    __test__.constrain(
      { kind: "var", name: "T" },
      { kind: "var", name: "T" },
      { kind: "edge", sourceNode: "a", sourceSlot: "out", targetNode: "b", targetSlot: "in" },
      sameVarBounds as never,
      [] as never,
      new Set(),
    );
    expect(sameVarBounds.size).toBe(0);

    expect(__test__.applySubst(new Map(), { kind: "var", name: "Unchanged" })).toEqual({
      kind: "var",
      name: "Unchanged",
    });

    expect(__test__.coalesce(new Map([
      ["VarOnlyUpper", { lower: [], upper: [{ kind: "var", name: "T" }] }],
    ]))).toEqual(new Map());
  });
});


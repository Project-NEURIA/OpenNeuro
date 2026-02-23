import type { Graph, SlotType } from "./types";

export type Type =
  | { kind: "concrete"; name: string }
  | { kind: "var"; name: string };

export type Origin =
  | { kind: "node_slot"; nodeId: string; direction: "in" | "out"; slot: string }
  | { kind: "edge"; sourceNode: string; sourceSlot: string; targetNode: string; targetSlot: string };

export interface Constraint {
  left: Type;
  right: Type;
  origin: Origin;
}

export interface TypeError {
  constraint: Constraint;
  left: Type;
  right: Type;
}

function parseType(s: string): Type {
  return { kind: "concrete", name: s };
}

export function getConstraints(
  graph: Graph,
  componentInputs: Record<string, Record<string, SlotType>>,
  componentOutputs: Record<string, Record<string, SlotType>>,
): Constraint[] {
  const constraints: Constraint[] = [];

  function slotVar(nodeId: string, direction: "in" | "out", slot: string): Type {
    return { kind: "var", name: `${nodeId}.${direction}.${slot}` };
  }

  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    const inputs = componentInputs[node.type] ?? {};
    for (const [slot, slotType] of Object.entries(inputs)) {
      constraints.push({
        left: slotVar(nodeId, "in", slot),
        right: parseType(slotType.name),
        origin: { kind: "node_slot", nodeId, direction: "in", slot },
      });
    }

    const outputs = componentOutputs[node.type] ?? {};
    for (const [slot, slotType] of Object.entries(outputs)) {
      constraints.push({
        left: slotVar(nodeId, "out", slot),
        right: parseType(slotType.name),
        origin: { kind: "node_slot", nodeId, direction: "out", slot },
      });
    }
  }

  for (const edge of graph.edges) {
    constraints.push({
      left: slotVar(edge.source_node, "out", edge.source_slot),
      right: slotVar(edge.target_node, "in", edge.target_slot),
      origin: {
        kind: "edge",
        sourceNode: edge.source_node,
        sourceSlot: edge.source_slot,
        targetNode: edge.target_node,
        targetSlot: edge.target_slot,
      },
    });
  }

  return constraints;
}

type Subst = Map<string, Type>;

function typeEquals(a: Type, b: Type): boolean {
  if (a.kind === "var" && b.kind === "var") return a.name === b.name;
  if (a.kind === "concrete" && b.kind === "concrete") return a.name === b.name;
  return false;
}

function occursIn(varName: string, t: Type): boolean {
  if (t.kind === "var") return t.name === varName;
  return false;
}

function applySubst(subst: Subst, t: Type): Type {
  if (t.kind === "var") {
    const replacement = subst.get(t.name);
    return replacement ? applySubst(subst, replacement) : t;
  }
  return t;
}

function applySubstToConstraints(subst: Subst, constraints: Constraint[]): Constraint[] {
  return constraints.map((c) => ({
    left: applySubst(subst, c.left),
    right: applySubst(subst, c.right),
    origin: c.origin,
  }));
}

function composeSubst(s1: Subst, s2: Subst): Subst {
  const result: Subst = new Map();
  for (const [k, v] of s1) {
    result.set(k, applySubst(s2, v));
  }
  for (const [k, v] of s2) {
    if (!result.has(k)) result.set(k, v);
  }
  return result;
}

interface UnifyResult {
  subst: Subst;
  errors: TypeError[];
}

export function unify(constraints: Constraint[]): UnifyResult {
  if (constraints.length === 0) return { subst: new Map(), errors: [] };

  const [first, ...rest] = constraints;
  const l = first!.left;
  const r = first!.right;

  if (typeEquals(l, r)) {
    return unify(rest);
  }

  if (l.kind === "concrete" && r.kind === "concrete") {
    const result = unify(rest);
    result.errors.push({ constraint: first!, left: l, right: r });
    return result;
  }

  if (l.kind === "var") {
    if (occursIn(l.name, r)) {
      const result = unify(rest);
      result.errors.push({ constraint: first!, left: l, right: r });
      return result;
    }
    const s: Subst = new Map([[l.name, r]]);
    const result = unify(applySubstToConstraints(s, rest));
    result.subst = composeSubst(s, result.subst);
    return result;
  }

  if (r.kind === "var") {
    if (occursIn(r.name, l)) {
      const result = unify(rest);
      result.errors.push({ constraint: first!, left: l, right: r });
      return result;
    }
    const s: Subst = new Map([[r.name, l]]);
    const result = unify(applySubstToConstraints(s, rest));
    result.subst = composeSubst(s, result.subst);
    return result;
  }

  const result = unify(rest);
  result.errors.push({ constraint: first!, left: l, right: r });
  return result;
}

export interface CheckResult {
  types: Map<string, Type>;
  errors: TypeError[];
}

export function typeToString(t: Type): string {
  return t.name;
}

export function checkTypes(
  graph: Graph,
  componentInputs: Record<string, Record<string, SlotType>>,
  componentOutputs: Record<string, Record<string, SlotType>>,
): CheckResult {
  const constraints = getConstraints(graph, componentInputs, componentOutputs);
  const { subst, errors } = unify(constraints);

  const types = new Map<string, Type>();
  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    const inputs = componentInputs[node.type] ?? {};
    for (const slot of Object.keys(inputs)) {
      const varName = `${nodeId}.in.${slot}`;
      types.set(varName, applySubst(subst, { kind: "var", name: varName }));
    }
    const outputs = componentOutputs[node.type] ?? {};
    for (const slot of Object.keys(outputs)) {
      const varName = `${nodeId}.out.${slot}`;
      types.set(varName, applySubst(subst, { kind: "var", name: varName }));
    }
  }

  return { types, errors };
}

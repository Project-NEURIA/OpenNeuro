import type { Graph } from "./types";

export type Type =
  | { kind: "concrete"; name: "int" | "str" | "Frame" }
  | { kind: "var"; name: string }
  | { kind: "channel"; inner: Type };

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

const CHANNEL_RE = /^(?:Channel|Sender|Receiver)\[(.+)\]$/;

function parseType(s: string): Type {
  const base = s.replace(/\s*\|\s*None$/, "").trim();

  const m = base.match(CHANNEL_RE);
  if (m) {
    return { kind: "channel", inner: parseType(m[1]!) };
  }

  if (base === "int" || base === "str" || base === "Frame") {
    return { kind: "concrete", name: base };
  }

  return { kind: "concrete", name: base as "int" | "str" | "Frame" };
}

export function getConstraints(
  graph: Graph,
  componentInputs: Record<string, Record<string, string>>,
  componentOutputs: Record<string, Record<string, string>>,
): Constraint[] {
  const constraints: Constraint[] = [];

  function slotVar(nodeId: string, direction: "in" | "out", slot: string): Type {
    return { kind: "var", name: `${nodeId}.${direction}.${slot}` };
  }

  for (const [nodeId, node] of Object.entries(graph.nodes)) {
    const inputs = componentInputs[node.type] ?? {};
    for (const [slot, typeStr] of Object.entries(inputs)) {
      constraints.push({
        left: slotVar(nodeId, "in", slot),
        right: parseType(typeStr),
        origin: { kind: "node_slot", nodeId, direction: "in", slot },
      });
    }

    const outputs = componentOutputs[node.type] ?? {};
    for (const [slot, typeStr] of Object.entries(outputs)) {
      constraints.push({
        left: slotVar(nodeId, "out", slot),
        right: parseType(typeStr),
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
  if (a.kind === "channel" && b.kind === "channel") return typeEquals(a.inner, b.inner);
  return false;
}

function occursIn(varName: string, t: Type): boolean {
  switch (t.kind) {
    case "var":
      return t.name === varName;
    case "concrete":
      return false;
    case "channel":
      return occursIn(varName, t.inner);
  }
}

function applySubst(subst: Subst, t: Type): Type {
  switch (t.kind) {
    case "var": {
      const replacement = subst.get(t.name);
      return replacement ? applySubst(subst, replacement) : t;
    }
    case "concrete":
      return t;
    case "channel":
      return { kind: "channel", inner: applySubst(subst, t.inner) };
  }
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

  // If already equal, delete and recurse
  if (typeEquals(l, r)) {
    return unify(rest);
  }

  // Both are constructors (concrete or channel) — decompose
  if (l.kind === "channel" && r.kind === "channel") {
    return unify([{ left: l.inner, right: r.inner, origin: first!.origin }, ...rest]);
  }

  if (l.kind === "concrete" && r.kind === "concrete") {
    const result = unify(rest);
    result.errors.push({ constraint: first!, left: l, right: r });
    return result;
  }

  // One is a variable — substitute
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

  // Mismatched constructors (e.g. concrete vs channel)
  const result = unify(rest);
  result.errors.push({ constraint: first!, left: l, right: r });
  return result;
}

export interface CheckResult {
  types: Map<string, Type>;
  errors: TypeError[];
}

export function typeToString(t: Type): string {
  switch (t.kind) {
    case "concrete":
      return t.name;
    case "var":
      return t.name;
    case "channel":
      return `Channel[${typeToString(t.inner)}]`;
  }
}

export function checkTypes(
  graph: Graph,
  componentInputs: Record<string, Record<string, string>>,
  componentOutputs: Record<string, Record<string, string>>,
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

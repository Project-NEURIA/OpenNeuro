import type { Graph, SlotType } from "./types";

export type Type =
  | { kind: "concrete"; name: string }
  | { kind: "var"; name: string }
  | { kind: "union"; types: Type[] }
  | { kind: "constructor"; name: string; inner: Type };

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

// LUB — least upper bound (smallest common supertype)
function join(_a: Type, _b: Type): Type | null {
  return null;
}

// GLB — greatest lower bound (largest common subtype)
function meet(_a: Type, _b: Type): Type | null {
  return null;
}

// a <= b?
function isSubtype(a: Type, b: Type): boolean {
  if (a.kind === "concrete" && b.kind === "concrete") {
    if (a.name === b.name) return true;
    const j = join(a, b);
    return j !== null && j.kind === "concrete" && j.name === b.name;
  }
  if (a.kind === "constructor" && b.kind === "constructor") {
    return a.name === b.name && isSubtype(a.inner, b.inner);
  }
  if (b.kind === "union") {
    return b.types.some((m) => isSubtype(a, m));
  }
  if (a.kind === "union") {
    return a.types.every((m) => isSubtype(m, b));
  }
  return false;
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

function occursIn(varName: string, t: Type): boolean {
  switch (t.kind) {
    case "var":
      return t.name === varName;
    case "concrete":
      return false;
    case "constructor":
      return occursIn(varName, t.inner);
    case "union":
      return t.types.some((m) => occursIn(varName, m));
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
    case "constructor":
      return { kind: "constructor", name: t.name, inner: applySubst(subst, t.inner) };
    case "union":
      return { kind: "union", types: t.types.map((m) => applySubst(subst, m)) };
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

function bindVar(varName: string, t: Type, first: Constraint, rest: Constraint[]): UnifyResult {
  if (occursIn(varName, t)) {
    const result = unify(rest);
    result.errors.push({ constraint: first, left: first.left, right: first.right });
    return result;
  }
  const s: Subst = new Map([[varName, t]]);
  const result = unify(applySubstToConstraints(s, rest));
  result.subst = composeSubst(s, result.subst);
  return result;
}

interface UnifyResult {
  subst: Subst;
  errors: TypeError[];
}

// Constraints are l <= r (left is subtype of right).
export function unify(constraints: Constraint[]): UnifyResult {
  if (constraints.length === 0) return { subst: new Map(), errors: [] };

  const [first, ...rest] = constraints;
  const l = first!.left;
  const r = first!.right;

  switch (l.kind) {
    case "concrete":
      switch (r.kind) {
        case "concrete": {
          const result = unify(rest);
          if (!isSubtype(l, r)) {
            result.errors.push({ constraint: first!, left: l, right: r });
          }
          return result;
        }
        case "var": {
          return bindVar(r.name, l, first!, rest);
        }
        case "constructor": {
          const result = unify(rest);
          result.errors.push({ constraint: first!, left: l, right: r });
          return result;
        }
        case "union": {
          const result = unify(rest);
          if (!isSubtype(l, r)) {
            result.errors.push({ constraint: first!, left: l, right: r });
          }
          return result;
        }
      }
    case "var":
      switch (r.kind) {
        case "concrete": {
          return bindVar(l.name, r, first!, rest);
        }
        case "var": {
          if (l.name === r.name) return unify(rest);
          return bindVar(l.name, r, first!, rest);
        }
        case "constructor": {
          return bindVar(l.name, r, first!, rest);
        }
        case "union": {
          return bindVar(l.name, r, first!, rest);
        }
      }
    case "constructor":
      switch (r.kind) {
        case "concrete": {
          const result = unify(rest);
          result.errors.push({ constraint: first!, left: l, right: r });
          return result;
        }
        case "var": {
          return bindVar(r.name, l, first!, rest);
        }
        case "constructor": {
          if (l.name !== r.name) {
            const result = unify(rest);
            result.errors.push({ constraint: first!, left: l, right: r });
            return result;
          }
          // covariant: l.inner <= r.inner
          return unify([{ left: l.inner, right: r.inner, origin: first!.origin }, ...rest]);
        }
        case "union": {
          const result = unify(rest);
          if (!isSubtype(l, r)) {
            result.errors.push({ constraint: first!, left: l, right: r });
          }
          return result;
        }
      }
    case "union":
      switch (r.kind) {
        case "concrete": {
          // every member of the union must be <= concrete
          const result = unify(rest);
          if (!isSubtype(l, r)) {
            result.errors.push({ constraint: first!, left: l, right: r });
          }
          return result;
        }
        case "var": {
          return bindVar(r.name, l, first!, rest);
        }
        case "constructor": {
          const result = unify(rest);
          if (!isSubtype(l, r)) {
            result.errors.push({ constraint: first!, left: l, right: r });
          }
          return result;
        }
        case "union": {
          // every member of l must be <= some member of r
          const result = unify(rest);
          if (!isSubtype(l, r)) {
            result.errors.push({ constraint: first!, left: l, right: r });
          }
          return result;
        }
      }
  }
}

export interface CheckResult {
  types: Map<string, Type>;
  errors: TypeError[];
}

export function typeToString(t: Type): string {
  switch (t.kind) {
    case "concrete":
    case "var":
      return t.name;
    case "constructor":
      return `${t.name}[${typeToString(t.inner)}]`;
    case "union":
      return t.types.map(typeToString).join(" | ");
  }
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

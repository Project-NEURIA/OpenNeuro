// Type inference with subtyping via eager constraint propagation.
// Based on: Lionel Parreaux, "The Simple Essence of Algebraic Subtyping:
// Principal Type Inference with Subtyping Made Easy", ICFP 2020.
// https://doi.org/10.1145/3409006

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

// --- Lattice ---

function flattenUnion(types: Type[]): Type[] {
  const result: Type[] = [];
  for (const t of types) {
    if (t.kind === "union") result.push(...flattenUnion(t.types));
    else result.push(t);
  }
  return result;
}

function typesEqual(a: Type, b: Type): boolean {
  if (a.kind === "concrete" && b.kind === "concrete") return a.name === b.name;
  if (a.kind === "var" && b.kind === "var") return a.name === b.name;
  if (a.kind === "constructor" && b.kind === "constructor") {
    return a.name === b.name && typesEqual(a.inner, b.inner);
  }
  if (a.kind === "union" && b.kind === "union") {
    if (a.types.length !== b.types.length) return false;
    return a.types.every((at) => b.types.some((bt) => typesEqual(at, bt)));
  }
  return false;
}

function dedup(types: Type[]): Type[] {
  const result: Type[] = [];
  for (const t of types) {
    if (!result.some((r) => typesEqual(r, t))) result.push(t);
  }
  return result;
}

// a <= b?
function isSubtype(a: Type, b: Type): boolean {
  if (typesEqual(a, b)) return true;
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

// LUB — least upper bound (smallest common supertype)
function join(a: Type, b: Type): Type {
  if (isSubtype(a, b)) return b;
  if (isSubtype(b, a)) return a;
  const members = dedup(flattenUnion([a, b]));
  if (members.length === 1) return members[0]!;
  return { kind: "union", types: members };
}

// --- Constraint generation ---

function splitTopLevel(s: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "[") depth++;
    else if (s[i] === "]") depth--;
    else if (s[i] === "," && depth === 0) {
      parts.push(s.slice(start, i).trim());
      start = i + 1;
    }
  }
  parts.push(s.slice(start).trim());
  return parts;
}

export function collectLeafNames(s: string): string[] {
  s = s.trim();
  const bracket = s.indexOf("[");
  if (bracket !== -1 && s.endsWith("]")) {
    const inner = s.slice(bracket + 1, -1);
    return splitTopLevel(inner).flatMap(collectLeafNames);
  }
  return [s];
}

function parseType(s: string, concreteTypes: Set<string>, scope?: string): Type {
  s = s.trim();
  const bracket = s.indexOf("[");
  if (bracket !== -1 && s.endsWith("]")) {
    const name = s.slice(0, bracket);
    const inner = s.slice(bracket + 1, -1);
    const args = splitTopLevel(inner);

    if (name === "Union") {
      const types = args.map((a) => parseType(a, concreteTypes, scope));
      return types.length === 1 ? types[0]! : { kind: "union", types };
    }

    if (args.length === 1) {
      return { kind: "constructor", name, inner: parseType(args[0]!, concreteTypes, scope) };
    }
  }
  if (concreteTypes.has(s)) return { kind: "concrete", name: s };
  return { kind: "var", name: scope ? `${scope}.${s}` : s };
}

export function getConstraints(
  graph: Graph,
  componentInputs: Record<string, Record<string, SlotType>>,
  componentOutputs: Record<string, Record<string, SlotType>>,
  concreteTypes: Set<string>,
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
        right: parseType(slotType.name, concreteTypes, nodeId),
        origin: { kind: "node_slot", nodeId, direction: "in", slot },
      });
    }

    const outputs = componentOutputs[node.type] ?? {};
    for (const [slot, slotType] of Object.entries(outputs)) {
      constraints.push({
        left: parseType(slotType.name, concreteTypes, nodeId),
        right: slotVar(nodeId, "out", slot),
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

// --- Simple-sub style constraint solving ---
// Mutable variable bounds with eager propagation.
// (Parreaux, "The Simple Essence of Algebraic Subtyping", 2020)

interface VarBounds {
  lower: Type[];
  upper: Type[];
}

type BoundsMap = Map<string, VarBounds>;

function getBounds(map: BoundsMap, name: string): VarBounds {
  let b = map.get(name);
  if (!b) {
    b = { lower: [], upper: [] };
    map.set(name, b);
  }
  return b;
}

function typeKey(t: Type): string {
  switch (t.kind) {
    case "concrete": return `c:${t.name}`;
    case "var": return `v:${t.name}`;
    case "constructor": return `k:${t.name}[${typeKey(t.inner)}]`;
    case "union": return `u:(${t.types.map(typeKey).sort().join("|")})`;
  }
}

/**
 * Enforce lhs <: rhs, eagerly propagating through variable bounds.
 *
 * Following Simple-sub's constrain():
 *   Variable(lhs) <: rhs  →  add rhs to upper bounds, propagate existing lower bounds against rhs
 *   lhs <: Variable(rhs)  →  add lhs to lower bounds, propagate lhs against existing upper bounds
 *   ground <: ground       →  check structurally (decompose constructors covariantly)
 */
function constrain(
  lhs: Type,
  rhs: Type,
  origin: Origin,
  bounds: BoundsMap,
  errors: TypeError[],
  cache: Set<string>,
): void {
  const key = typeKey(lhs) + "<:" + typeKey(rhs);
  if (cache.has(key)) return;
  cache.add(key);

  // --- Variable on the left: lhs <: rhs → upper bound on lhs ---
  if (lhs.kind === "var") {
    if (rhs.kind === "var" && lhs.name === rhs.name) return;
    const b = getBounds(bounds, lhs.name);
    b.upper.push(rhs);
    for (const lb of [...b.lower]) {
      constrain(lb, rhs, origin, bounds, errors, cache);
    }
    return;
  }

  // --- Variable on the right: lhs <: rhs → lower bound on rhs ---
  if (rhs.kind === "var") {
    const b = getBounds(bounds, rhs.name);
    b.lower.push(lhs);
    for (const ub of [...b.upper]) {
      constrain(lhs, ub, origin, bounds, errors, cache);
    }
    return;
  }

  // --- Ground cases (no variables) ---

  // Constructor <: Constructor → covariant decomposition
  if (lhs.kind === "constructor" && rhs.kind === "constructor") {
    if (lhs.name === rhs.name) {
      constrain(lhs.inner, rhs.inner, origin, bounds, errors, cache);
    } else {
      errors.push({ constraint: { left: lhs, right: rhs, origin }, left: lhs, right: rhs });
    }
    return;
  }

  // Everything else → direct subtype check
  if (!isSubtype(lhs, rhs)) {
    errors.push({ constraint: { left: lhs, right: rhs, origin }, left: lhs, right: rhs });
  }
}

// --- Coalescing: read off resolved types from variable bounds ---

type Subst = Map<string, Type>;

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

function coalesce(bounds: BoundsMap): Subst {
  const subst: Subst = new Map();
  for (const [varName, b] of bounds) {
    const concreteLower = b.lower.filter((t) => t.kind !== "var");
    if (concreteLower.length > 0) {
      let resolved = concreteLower[0]!;
      for (let i = 1; i < concreteLower.length; i++) {
        resolved = join(resolved, concreteLower[i]!);
      }
      subst.set(varName, resolved);
    } else {
      const concreteUpper = b.upper.filter((t) => t.kind !== "var");
      if (concreteUpper.length > 0) {
        subst.set(varName, concreteUpper[0]!);
      }
    }
  }
  return subst;
}

// --- Public API ---

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
  concreteTypes: Set<string>,
): CheckResult {
  const constraints = getConstraints(graph, componentInputs, componentOutputs, concreteTypes);

  const bounds: BoundsMap = new Map();
  const errors: TypeError[] = [];
  const cache = new Set<string>();

  for (const c of constraints) {
    constrain(c.left, c.right, c.origin, bounds, errors, cache);
  }

  const subst = coalesce(bounds);

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

interface PositionedNode {
  id: string;
  x: number;
  y: number;
}

/**
 * Topological sort of pipeline nodes, then assign left-to-right positions.
 */
export function layoutNodes(
  nodes: { id: string }[],
  edges: { source: string; target: string }[],
): PositionedNode[] {
  const adj = new Map<string, string[]>();
  const inDeg = new Map<string, number>();

  for (const n of nodes) {
    adj.set(n.id, []);
    inDeg.set(n.id, 0);
  }

  for (const e of edges) {
    adj.get(e.source)?.push(e.target);
    inDeg.set(e.target, (inDeg.get(e.target) ?? 0) + 1);
  }

  // Kahn's algorithm
  const queue: string[] = [];
  for (const [id, deg] of inDeg) {
    if (deg === 0) queue.push(id);
  }

  const sorted: string[] = [];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    sorted.push(cur);
    for (const next of adj.get(cur) ?? []) {
      const d = (inDeg.get(next) ?? 1) - 1;
      inDeg.set(next, d);
      if (d === 0) queue.push(next);
    }
  }

  // Fallback: add any nodes not in sorted (disconnected)
  for (const n of nodes) {
    if (!sorted.includes(n.id)) sorted.push(n.id);
  }

  const xGap = 280;
  const yCenter = 200;
  const yWave = 30;

  return sorted.map((id, i) => ({
    id,
    x: 80 + i * xGap,
    y: yCenter + Math.sin(i * 0.8) * yWave,
  }));
}

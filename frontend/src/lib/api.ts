import type { ComponentInfo } from "./types";

export async function fetchComponents(): Promise<ComponentInfo[]> {
  const res = await fetch("/component");
  if (!res.ok) throw new Error(`Fetch components failed: ${res.status}`);
  return res.json();
}

export async function fetchNodes(): Promise<{ id: string; type: string; status: string }[]> {
  const res = await fetch("/graph/nodes");
  if (!res.ok) throw new Error(`Fetch nodes failed: ${res.status}`);
  return res.json();
}

export async function fetchEdges(): Promise<{ source: string; target: string }[]> {
  const res = await fetch("/graph/edges");
  if (!res.ok) throw new Error(`Fetch edges failed: ${res.status}`);
  return res.json();
}

export async function createNode(type: string, id?: string) {
  const res = await fetch("/graph/nodes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, id }),
  });
  if (!res.ok) throw new Error(`Create node failed: ${res.status}`);
  return res.json() as Promise<{ id: string; type: string; status: string }>;
}

export async function deleteNode(id: string) {
  const res = await fetch(`/graph/nodes/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete node failed: ${res.status}`);
}

export async function createEdge(source: string, target: string) {
  const res = await fetch("/graph/edges", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, target }),
  });
  if (!res.ok) throw new Error(`Create edge failed: ${res.status}`);
  return res.json() as Promise<{ source: string; target: string }>;
}

export async function deleteEdge(source: string, target: string) {
  const res = await fetch("/graph/edges", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, target }),
  });
  if (!res.ok) throw new Error(`Delete edge failed: ${res.status}`);
}

export async function startAll() {
  const res = await fetch("/graph/start", { method: "POST" });
  if (!res.ok) throw new Error(`Start failed: ${res.status}`);
}

export async function stopAll() {
  const res = await fetch("/graph/stop", { method: "POST" });
  if (!res.ok) throw new Error(`Stop failed: ${res.status}`);
}

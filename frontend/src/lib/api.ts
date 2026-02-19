import type { ComponentInfo } from "./types";

export interface EdgeData {
  source_node: string;
  source_slot: string;
  target_node: string;
  target_slot: string;
}

export async function fetchComponents(): Promise<ComponentInfo[]> {
  const res = await fetch("/component");
  if (!res.ok) throw new Error(`Fetch components failed: ${res.status}`);
  return res.json();
}

export async function fetchNodes(): Promise<
  { id: string; type: string; status: string; x: number; y: number }[]
> {
  const res = await fetch("/graph/nodes");
  if (!res.ok) throw new Error(`Fetch nodes failed: ${res.status}`);
  return res.json();
}

export async function fetchEdges(): Promise<EdgeData[]> {
  const res = await fetch("/graph/edges");
  if (!res.ok) throw new Error(`Fetch edges failed: ${res.status}`);
  return res.json();
}

export async function createNode(type: string, init_args?: Record<string, unknown>) {
  const res = await fetch("/graph/nodes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, init_args }),
  });
  if (!res.ok) throw new Error(`Create node failed: ${res.status}`);
  return res.json() as Promise<{
    id: string;
    type: string;
    status: string;
    x: number;
    y: number;
  }>;
}

export async function updateNode(id: string, data: { x?: number; y?: number }) {
  const res = await fetch(`/graph/nodes/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Update node failed: ${res.status}`);
}

export async function deleteNode(id: string) {
  const res = await fetch(`/graph/nodes/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete node failed: ${res.status}`);
}

export async function createEdge(
  source_node: string,
  source_slot: string,
  target_node: string,
  target_slot: string,
) {
  const res = await fetch("/graph/edges", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_node, source_slot, target_node, target_slot }),
  });
  if (!res.ok) throw new Error(`Create edge failed: ${res.status}`);
  return res.json() as Promise<EdgeData>;
}

export async function deleteEdge(
  source_node: string,
  source_slot: string,
  target_node: string,
  target_slot: string,
) {
  const res = await fetch("/graph/edges", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_node, source_slot, target_node, target_slot }),
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

// --- Project API ---

export interface ProjectSummary {
  name: string;
  has_thumbnail: boolean;
}

export async function fetchProjects(): Promise<ProjectSummary[]> {
  const res = await fetch("/projects");
  if (!res.ok) throw new Error(`Fetch projects failed: ${res.status}`);
  return res.json();
}

export async function createProject(name: string) {
  const res = await fetch("/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(`Create project failed: ${res.status}`);
}

export async function deleteProject(name: string) {
  const res = await fetch(`/projects/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Delete project failed: ${res.status}`);
}

export async function startProject(
  name: string,
): Promise<{ current_project: string | null }> {
  const res = await fetch(`/projects/${encodeURIComponent(name)}/start`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Start project failed: ${res.status}`);
  return res.json();
}

export async function closeProject() {
  const res = await fetch("/project/close", { method: "POST" });
  if (!res.ok) throw new Error(`Close project failed: ${res.status}`);
}

export async function fetchCurrentProject(): Promise<{
  current_project: string | null;
}> {
  const res = await fetch("/project/current");
  if (!res.ok) throw new Error(`Fetch current project failed: ${res.status}`);
  return res.json();
}

export function projectThumbnailUrl(name: string): string {
  return `/projects/${encodeURIComponent(name)}/thumbnail`;
}

export async function saveGraph() {
  const res = await fetch("/graph/save", { method: "POST" });
  if (!res.ok) throw new Error(`Save graph failed: ${res.status}`);
}

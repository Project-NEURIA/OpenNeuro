import type { PipelineTopology } from "./types";

export async function fetchPipeline(): Promise<PipelineTopology> {
  const res = await fetch("/api/pipeline");
  if (!res.ok) throw new Error(`Pipeline fetch failed: ${res.status}`);
  return res.json();
}

export async function postPipeline(config: {
  nodes: string[];
  edges: { source: string; target: string }[];
}): Promise<{ status: string; detail?: string }> {
  const res = await fetch("/api/pipeline", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`Pipeline POST failed: ${res.status}`);
  return res.json();
}

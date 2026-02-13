import { memo } from "react";
import {
  BaseEdge,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";

/** Map byte throughput delta to a color from cold (idle) to hot (busy). */
function throughputColor(byteDelta: number): string {
  if (byteDelta === 0) return "var(--edge)";
  // Normalize: ~10 KB/tick = full intensity
  const t = Math.min(byteDelta / 10_000, 1);
  // Interpolate hue from 120 (green/idle) → 0 (red/hot)
  const hue = Math.round(120 * (1 - t));
  const lightness = 45 + t * 15;
  return `hsl(${hue} 80% ${lightness}%)`;
}

function PipelineEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const byteDelta = (data as Record<string, unknown>)?.byteDelta as number ?? 0;
  const color = throughputColor(byteDelta);
  const thickness = Math.min(1.5 + (byteDelta / 10_000) * 2.5, 4);

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      style={{
        stroke: color,
        strokeWidth: thickness,
        transition: "stroke 0.3s ease, stroke-width 0.3s ease",
      }}
    />
  );
}

export const PipelineEdge = memo(PipelineEdgeComponent);

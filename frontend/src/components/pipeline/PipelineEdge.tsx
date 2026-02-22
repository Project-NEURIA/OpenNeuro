import { memo } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
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
  const typeError = (data as Record<string, unknown>)?.typeError as string | undefined;

  const color = typeError ? "hsl(0 80% 50%)" : throughputColor(byteDelta);
  const thickness = typeError ? 2.5 : Math.min(1.5 + (byteDelta / 10_000) * 2.5, 4);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: color,
          strokeWidth: thickness,
          transition: "stroke 0.3s ease, stroke-width 0.3s ease",
        }}
      />
      {typeError && (
        <EdgeLabelRenderer>
          <div
            className="nodrag nopan"
            style={{
              position: "absolute",
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
              pointerEvents: "all",
              background: "hsl(0 60% 15%)",
              color: "hsl(0 80% 70%)",
              border: "1px solid hsl(0 60% 30%)",
              borderRadius: "4px",
              padding: "2px 6px",
              fontSize: "11px",
              fontFamily: "monospace",
              whiteSpace: "nowrap",
            }}
          >
            {typeError}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export const PipelineEdge = memo(PipelineEdgeComponent);

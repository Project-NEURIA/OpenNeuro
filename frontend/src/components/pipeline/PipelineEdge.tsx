import { memo } from "react";
import {
  BaseEdge,
  getBezierPath,
  type EdgeProps,
} from "@xyflow/react";

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
  const msgPerSec = (data as Record<string, unknown>)?.msgPerSec as number ?? 0;
  const thickness = Math.min(1.5 + msgPerSec * 0.3, 5);
  const opacity = Math.min(0.3 + msgPerSec * 0.07, 1);

  const [edgePath] = getBezierPath({
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
          stroke: `rgba(113, 113, 122, ${opacity})`,
          strokeWidth: thickness,
        }}
      />
      {/* Animated flowing dot */}
      <circle r={3} fill="#a1a1aa" opacity={0.8}>
        <animateMotion dur={`${Math.max(3 - msgPerSec * 0.2, 0.5)}s`} repeatCount="indefinite" path={edgePath} />
      </circle>
    </>
  );
}

export const PipelineEdge = memo(PipelineEdgeComponent);

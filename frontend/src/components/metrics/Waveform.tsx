import { memo, useId } from "react";

interface WaveformProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  className?: string;
  showAxes?: boolean;
  formatY?: (v: number) => string;
  duration?: number; // total seconds the data window spans
}

const LABEL_COLOR = "rgba(255,255,255,0.4)";
const GRID_COLOR = "rgba(255,255,255,0.06)";

function WaveformComponent({
  data,
  width = 200,
  height = 32,
  color = "#4ade80",
  className,
  showAxes = false,
  formatY,
  duration,
}: WaveformProps) {
  const uid = useId().replace(/:/g, "");
  const gradientId = `wf-grad-${uid}`;
  const glowId = `wf-glow-${uid}`;

  const pad = 1;
  const marginB = showAxes ? 14 : pad;

  const plotW = width - pad * 2;
  const plotH = height - pad - marginB;

  const max = Math.max(...data, 1);

  const points = data.map((v, i) => {
    const x = pad + (data.length > 1 ? (i / (data.length - 1)) * plotW : plotW);
    const y = pad + plotH - (v / max) * plotH;
    return `${x},${y}`;
  });

  const polyline = points.join(" ");
  const lastPoint = points[points.length - 1];
  const [lastX, lastY] = lastPoint ? lastPoint.split(",").map(Number) : [0, 0];

  const baseline = pad + plotH;
  const areaPath =
    points.length > 0
      ? `M${points[0]} ${points.slice(1).map((p) => `L${p}`).join(" ")} L${lastX},${baseline} L${pad},${baseline} Z`
      : "";

  // Y-axis ticks (rendered inside the plot area)
  const yFmt = formatY ?? ((v: number) => String(Math.round(v)));
  const yTicks = showAxes
    ? [
        { value: max, y: pad + 2, grid: true },
        { value: max / 2, y: pad + plotH / 2, grid: true },
        { value: 0, y: baseline - 2, grid: false },
      ]
    : [];

  // X-axis ticks
  const xTicks: { label: string; x: number }[] = [];
  if (showAxes && duration && duration > 0) {
    const steps = [0, 0.5, 1];
    for (const s of steps) {
      const secsAgo = duration * (1 - s);
      const label = secsAgo === 0 ? "now" : `-${secsAgo.toFixed(0)}s`;
      xTicks.push({ label, x: pad + s * plotW });
    }
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={className}
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
        <filter id={glowId}>
          <feGaussianBlur stdDeviation="3" result="blur" />
        </filter>
      </defs>

      {/* Grid lines */}
      {showAxes &&
        yTicks.filter((t) => t.grid).map((t, i) => (
          <line
            key={i}
            x1={pad}
            y1={t.y}
            x2={width - pad}
            y2={t.y}
            stroke={GRID_COLOR}
            strokeWidth={1}
          />
        ))}

      {/* Y-axis labels (inside, top-left) */}
      {yTicks.map((t, i) => (
        <text
          key={i}
          x={pad + 4}
          y={t.y}
          textAnchor="start"
          dominantBaseline={i === 0 ? "hanging" : i === yTicks.length - 1 ? "auto" : "central"}
          fill={LABEL_COLOR}
          fontSize={8}
          fontFamily="monospace"
        >
          {yFmt(t.value)}
        </text>
      ))}

      {/* X-axis labels */}
      {xTicks.map((t, i) => (
        <text
          key={i}
          x={t.x}
          y={height - 2}
          textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}
          fill={LABEL_COLOR}
          fontSize={8}
          fontFamily="monospace"
        >
          {t.label}
        </text>
      ))}

      {/* Bottom axis line */}
      {showAxes && (
        <line x1={pad} y1={baseline} x2={width - pad} y2={baseline} stroke={GRID_COLOR} strokeWidth={1} />
      )}

      {/* Area fill */}
      {areaPath && <path d={areaPath} fill={`url(#${gradientId})`} />}

      {/* Glow line */}
      <polyline
        points={polyline}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
        filter={`url(#${glowId})`}
        opacity={0.5}
      />

      {/* Main line */}
      <polyline
        points={polyline}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* Current value dot */}
      {lastPoint && (
        <circle cx={lastX} cy={lastY} r={2} fill={color} filter={`url(#${glowId})`} />
      )}
    </svg>
  );
}

export const Waveform = memo(WaveformComponent);

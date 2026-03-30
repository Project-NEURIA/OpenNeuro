import { useEffect, useRef } from "react";

const TEXT = "OpenNeuro";
const DRAW_DURATION = 1.3;
const FILL_DURATION = 0.2;
const OVERLAP = 1.05;

export function populateSplash(svg: SVGSVGElement | null) {
  if (!svg) return;

  const ns = "http://www.w3.org/2000/svg";

  // Measure total width
  const measure = document.createElementNS(ns, "text");
  measure.setAttribute("y", "95");
  measure.style.fontFamily = "'Helvetica Neue', Arial, sans-serif";
  measure.style.fontWeight = "700";
  measure.style.fontSize = "96px";
  measure.textContent = TEXT;
  svg.appendChild(measure);
  const totalWidth = measure.getComputedTextLength();
  svg.removeChild(measure);

  svg.setAttribute("viewBox", `0 0 ${totalWidth + 20} 120`);
  svg.setAttribute("width", String(totalWidth + 20));
  svg.setAttribute("height", "120");

  // Measure each letter's x position
  const positions: number[] = [];
  for (let i = 0; i < TEXT.length; i++) {
    const prev = document.createElementNS(ns, "text");
    prev.setAttribute("y", "95");
    prev.style.fontFamily = "'Helvetica Neue', Arial, sans-serif";
    prev.style.fontWeight = "700";
    prev.style.fontSize = "96px";
    prev.textContent = TEXT.substring(0, i);
    svg.appendChild(prev);
    positions.push(prev.getComputedTextLength() + 10);
    svg.removeChild(prev);
  }

  let delay = 0.3;

  // Stroke pass
  for (let i = 0; i < TEXT.length; i++) {
    const stroke = document.createElementNS(ns, "text");
    stroke.setAttribute("x", String(positions[i]));
    stroke.setAttribute("y", "95");
    stroke.setAttribute("class", "letter-stroke");
    if (TEXT[i] === "O") stroke.style.stroke = "#ff2d2d";
    stroke.textContent = TEXT[i];
    svg.appendChild(stroke);

    const len = 800;
    stroke.style.strokeDasharray = String(len);
    stroke.style.strokeDashoffset = String(len);
    stroke.style.animation = `draw ${DRAW_DURATION}s ease-out forwards`;
    stroke.style.animationDelay = `${delay}s`;

    delay += DRAW_DURATION - OVERLAP;
  }

  // Fill pass
  delay = 0.3;
  for (let i = 0; i < TEXT.length; i++) {
    const fill = document.createElementNS(ns, "text");
    fill.setAttribute("x", String(positions[i]));
    fill.setAttribute("y", "95");
    fill.setAttribute("class", "letter-fill");
    if (TEXT[i] === "O") fill.style.fill = "#ff2d2d";
    fill.textContent = TEXT[i];
    fill.style.animation = `fadeIn ${FILL_DURATION}s ease forwards`;
    fill.style.animationDelay = `${delay + DRAW_DURATION * 0.2}s`;
    svg.appendChild(fill);

    delay += DRAW_DURATION - OVERLAP;
  }
}

export function SplashScreen({ status }: { status: string }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    populateSplash(svgRef.current);
  }, []);

  return (
    <div className="relative flex h-screen w-screen flex-col items-center justify-center bg-[#0a0a0f]">
      <div data-tauri-drag-region className="absolute top-0 left-0 right-0 h-[15px]" />
      <style>{`
        .letter-stroke {
          font-family: 'Helvetica Neue', Arial, sans-serif;
          font-weight: 700;
          font-size: 96px;
          fill: none;
          stroke: #fff;
          stroke-width: 1.5;
        }
        .letter-fill {
          font-family: 'Helvetica Neue', Arial, sans-serif;
          font-weight: 700;
          font-size: 96px;
          fill: #fff;
          stroke: none;
          opacity: 0;
        }
        @keyframes draw {
          to { stroke-dashoffset: 0; }
        }
        @keyframes fadeIn {
          to { opacity: 1; }
        }
      `}</style>
      <svg ref={svgRef} id="logo" xmlns="http://www.w3.org/2000/svg" style={{ maxWidth: "90vw" }} />
      <div className="mt-8 h-[3px] w-[300px] overflow-hidden rounded-sm bg-white/[0.08]">
        <div className="h-full animate-pulse rounded-sm bg-[#ff2d2d]" style={{ width: "60%" }} />
      </div>
      <div className="mt-3 font-['Helvetica_Neue',Arial,sans-serif] text-xs tracking-wide text-white/35">
        {status}
      </div>
    </div>
  );
}

import { useSSE } from "@/hooks/useSSE";
import { useState, useEffect } from "react";
import { Search } from "lucide-react";
import type { FramesSnapshot } from "@/lib/types";

function formatPts(pts: number, startTime: number): string {
  // Convert nanoseconds to seconds relative to start time
  const relativeMs = (pts - startTime) / 1_000_000;
  const relativeSec = relativeMs / 1000;
  return `${relativeSec.toFixed(3)}s`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export function FramesInspector() {
  const { data: framesSnapshot, connected } = useSSE<FramesSnapshot>("/frames");
  const [pipelineStartTime, setPipelineStartTime] = useState<number | null>(null);
  const [originalStartTime, setOriginalStartTime] = useState<number | null>(null);
  const [lastResetTime, setLastResetTime] = useState<number>(Date.now());
  const [filterText, setFilterText] = useState<string>("");

  const frames = framesSnapshot?.frames ?? [];

  // Filter frames based on filter text
  const filteredFrames = frames.filter(frame => 
    filterText === "" || 
    frame.frame_type_string.toLowerCase().includes(filterText.toLowerCase()) ||
    frame.id.toString().includes(filterText) ||
    frame.message.toLowerCase().includes(filterText.toLowerCase())
  );

  // Listen for pipeline start/stop events
  useEffect(() => {
    const handleStart = () => {
      const startTime = Date.now() * 1_000_000; // Convert to nanoseconds
      setPipelineStartTime(startTime);
      setOriginalStartTime(startTime);
      setLastResetTime(Date.now());
    };

    const handleStop = () => {
      // Keep original start time for display, clear current running time
      setPipelineStartTime(null);
      setLastResetTime(Date.now());
    };

    // Listen for custom events from MetricsOverlay
    window.addEventListener('pipeline-start', handleStart);
    window.addEventListener('pipeline-stop', handleStop);

    return () => {
      window.removeEventListener('pipeline-start', handleStart);
      window.removeEventListener('pipeline-stop', handleStop);
    };
  }, []);

  // Reset frames when pipeline stops
  useEffect(() => {
    if (!pipelineStartTime && frames.length > 0) {
      // Frames will be cleared on next backend update
    }
  }, [pipelineStartTime, frames.length]);

  return (
    <div className="w-64 shrink-0 border-l border-zinc-800 bg-zinc-950 p-4 flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
          Frames Inspector
        </h2>
        <div className="flex items-center gap-2">
          <div
            className={`w-1.5 h-1.5 rounded-full ${
              connected ? "bg-green-400" : "bg-red-500"
            }`}
          />
          {pipelineStartTime && (
            <span className="text-[9px] text-emerald-400 font-mono">RUNNING</span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between text-[10px] text-zinc-600 mb-2 px-1">
        <span>Count: {filteredFrames.length}/{frames.length}/100</span>
        {pipelineStartTime && (
          <span className="text-zinc-500">
            Since: {new Date(pipelineStartTime / 1_000_000).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Filter Bar */}
      <div className="mb-3">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 w-3 h-3 text-zinc-500" />
          <input
            type="text"
            placeholder="Filter frames..."
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            className="w-full pl-7 pr-2 py-1.5 text-[11px] bg-zinc-900/50 border border-zinc-800/50 rounded text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-zinc-700"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-1">
        {filteredFrames.map((frame) => (
          <div
            key={`${frame.id}-${lastResetTime}`} // Force re-render on reset
            className="px-2 py-1.5 rounded bg-zinc-900/50 border border-zinc-800/50 text-[11px]"
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-zinc-400">#{frame.id}</span>
              <span className="text-zinc-500">{formatBytes(frame.size_bytes)}</span>
            </div>
            <div className="flex items-center justify-between mt-0.5">
              <span className="text-zinc-300">{frame.frame_type_string}</span>
              <span className="text-zinc-600 font-mono">
                {pipelineStartTime ? formatPts(frame.pts, pipelineStartTime) : 
                 originalStartTime ? formatPts(frame.pts, originalStartTime) : '--'}
              </span>
            </div>
            
            {/* Always show message section */}
            <div className="mt-2 pt-2 border-t border-zinc-800/50">
              <div className="text-[10px] text-zinc-500 font-mono break-all">
                {frame.message}
              </div>
            </div>
          </div>
        ))}

        {filteredFrames.length === 0 && (
          <div className="text-center text-zinc-600 text-[11px] py-8">
            {filterText ? 'No frames match filter' : 'No frames yet'}
          </div>
        )}
      </div>
    </div>
  );
}

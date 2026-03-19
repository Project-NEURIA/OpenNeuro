import { useCallback, useEffect, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  type EdgeTypes,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type OnNodeDrag,
  type OnSelectionChangeFunc,
  type Viewport,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { GraphNode } from "./GraphNode";
import { GraphEdge } from "./GraphEdge";
import { ConfiguringNode } from "./ConfiguringNode";

const nodeTypes: NodeTypes = {
  graph: GraphNode,
  configuring: ConfiguringNode,
};

const edgeTypes: EdgeTypes = {
  graph: GraphEdge,
};

interface GraphCanvasProps {
  nodes: Node[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect?: OnConnect;
  onDrop?: (event: React.DragEvent) => void;
  onDragOver?: (event: React.DragEvent) => void;
  onNodeDragStop?: OnNodeDrag;
  onSelectionChange?: OnSelectionChangeFunc;
  onNodeContextMenu?: (event: React.MouseEvent, node: Node) => void;
  onPaneClick?: () => void;
}

export function GraphCanvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onDrop,
  onDragOver,
  onNodeDragStop,
  onSelectionChange,
  onNodeContextMenu,
  onPaneClick,
}: GraphCanvasProps) {
  const [shiftHeld, setShiftHeld] = useState(false);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "Shift") setShiftHeld(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === "Shift") setShiftHeld(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.3}
        maxZoom={2}
        onViewportChange={(viewport: Viewport) => {
          // Round viewport to prevent subpixel blur in WebKit
          const el = document.querySelector('.react-flow__viewport') as HTMLElement | null;
          if (el) {
            const x = Math.round(viewport.x);
            const y = Math.round(viewport.y);
            el.style.transform = `translate(${x}px, ${y}px) scale(${viewport.zoom})`;
          }
        }}
        deleteKeyCode="Backspace"
        proOptions={{ hideAttribution: true }}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onNodeDragStop={onNodeDragStop}
        onSelectionChange={onSelectionChange}
        onNodeContextMenu={onNodeContextMenu}
        onPaneClick={onPaneClick}
        selectionOnDrag={shiftHeld}
        panOnDrag={shiftHeld ? false : [0]}
        selectionKeyCode={null}
        multiSelectionKeyCode="Shift"
      >
        <Background color="var(--grid)" gap={40} size={3} />
        <Controls
          className="!bg-secondary !border-border !rounded-lg [&>button]:!bg-secondary [&>button]:!border-border [&>button]:!text-muted-foreground [&>button:hover]:!bg-accent"
        />
      </ReactFlow>
    </div>
  );
}

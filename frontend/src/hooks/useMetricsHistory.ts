import { useEffect, useRef, useState } from "react";
import type { MetricsSnapshot } from "@/lib/types";

export interface SenderHistory {
  msgThroughput: number[];
  byteThroughput: number[];
  bufferDepths: number[];
}

export interface ReceiverHistory {
  msgThroughput: number[];
  byteThroughput: number[];
}

export interface NodeHistoryEntry {
  msgThroughput: number[];
  byteThroughput: number[];
  senderHistory: Record<string, SenderHistory>;
  receiverHistory: Record<string, ReceiverHistory>;
}

export interface MetricsHistory {
  current: MetricsSnapshot | null;
  snapshots: MetricsSnapshot[];
  snapshotRate: number;
  dt: number;
  nodeHistory: Record<string, NodeHistoryEntry>;
}

const MAX_LENGTH = 60;

export function useMetricsHistory(snapshot: MetricsSnapshot | null): MetricsHistory {
  const bufferRef = useRef<MetricsSnapshot[]>([]);
  const dtRef = useRef<number[]>([]);
  const lastRef = useRef<MetricsSnapshot | null>(null);
  const [history, setHistory] = useState<MetricsHistory>({
    current: null,
    snapshots: [],
    snapshotRate: 0,
    dt: 0,
    nodeHistory: {},
  });

  useEffect(() => {
    if (!snapshot || snapshot === lastRef.current) return;
    lastRef.current = snapshot;

    const buf = bufferRef.current;
    const dts = dtRef.current;

    const prev = buf[buf.length - 1];
    const dt = prev ? snapshot.timestamp - prev.timestamp : 0;

    buf.push(snapshot);
    dts.push(dt);
    if (buf.length > MAX_LENGTH) {
      buf.shift();
      dts.shift();
    }

    const snapshots = [...buf];
    const first = snapshots[0]!;
    const last = snapshots[snapshots.length - 1]!;
    const elapsed = last.timestamp - first.timestamp;
    const snapshotRate = elapsed > 0 ? snapshots.length / elapsed : 0;

    const nodeHistory: Record<string, NodeHistoryEntry> = {};

    const nodeIds = new Set<string>();
    for (const s of snapshots) {
      for (const id of Object.keys(s.nodes)) nodeIds.add(id);
    }

    for (const nodeId of nodeIds) {
      const msgThroughput: number[] = [];
      const byteThroughput: number[] = [];

      // Collect all sender/receiver slot names across snapshots
      const senderSlots = new Set<string>();
      const receiverSlots = new Set<string>();
      for (const s of snapshots) {
        const n = s.nodes[nodeId];
        if (n) {
          for (const slot of Object.keys(n.senders)) senderSlots.add(slot);
          for (const slot of Object.keys(n.receivers)) receiverSlots.add(slot);
        }
      }

      const senderHistory: Record<string, SenderHistory> = {};
      for (const slot of senderSlots) {
        senderHistory[slot] = { msgThroughput: [], byteThroughput: [], bufferDepths: [] };
      }

      const receiverHistory: Record<string, ReceiverHistory> = {};
      for (const slot of receiverSlots) {
        receiverHistory[slot] = { msgThroughput: [], byteThroughput: [] };
      }

      for (let i = 0; i < snapshots.length; i++) {
        const s = snapshots[i]!;
        const d = dts[i]!;
        const rate = d > 0 ? 1 / d : 0;
        const n = s.nodes[nodeId];

        if (!n) {
          msgThroughput.push(0);
          byteThroughput.push(0);
          for (const slot of senderSlots) {
            senderHistory[slot]!.msgThroughput.push(0);
            senderHistory[slot]!.byteThroughput.push(0);
            senderHistory[slot]!.bufferDepths.push(0);
          }
          for (const slot of receiverSlots) {
            receiverHistory[slot]!.msgThroughput.push(0);
            receiverHistory[slot]!.byteThroughput.push(0);
          }
          continue;
        }

        // Aggregate across all senders for node-level throughput
        let totalMsg = 0;
        let totalBytes = 0;
        for (const sender of Object.values(n.senders)) {
          totalMsg += sender.msg_count_delta;
          totalBytes += sender.byte_count_delta;
        }
        msgThroughput.push(totalMsg * rate);
        byteThroughput.push(totalBytes * rate);

        // Per-sender history
        for (const slot of senderSlots) {
          const sender = n.senders[slot];
          senderHistory[slot]!.msgThroughput.push((sender?.msg_count_delta ?? 0) * rate);
          senderHistory[slot]!.byteThroughput.push((sender?.byte_count_delta ?? 0) * rate);
          senderHistory[slot]!.bufferDepths.push(sender?.buffer_depth ?? 0);
        }

        // Per-receiver history
        for (const slot of receiverSlots) {
          const recv = n.receivers[slot];
          receiverHistory[slot]!.msgThroughput.push((recv?.msg_count_delta ?? 0) * rate);
          receiverHistory[slot]!.byteThroughput.push((recv?.byte_count_delta ?? 0) * rate);
        }
      }

      nodeHistory[nodeId] = { msgThroughput, byteThroughput, senderHistory, receiverHistory };
    }

    setHistory({ current: snapshot, snapshots, snapshotRate, dt, nodeHistory });
  }, [snapshot]);

  return history;
}

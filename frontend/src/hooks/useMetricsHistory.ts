import { useEffect, useRef, useState } from "react";
import type { MetricsSnapshot } from "@/lib/types";

export interface SubscriberHistory {
  msgThroughput: number[];
  byteThroughput: number[];
}

export interface ChannelHistory {
  msgThroughput: number[];
  byteThroughput: number[];
  bufferDepths: number[];
  subscriberHistory: Record<string, SubscriberHistory>;
}

export interface NodeHistoryEntry {
  msgThroughput: number[];
  byteThroughput: number[];
  channelHistory: Record<string, ChannelHistory>;
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
      const channelIds = new Set<string>();

      for (const s of snapshots) {
        const n = s.nodes[nodeId];
        if (n) {
          for (const chId of Object.keys(n.channels)) channelIds.add(chId);
        }
      }

      // Collect all subscriber IDs per channel across all snapshots
      const channelSubIds: Record<string, Set<string>> = {};
      for (const chId of channelIds) {
        const subIds = new Set<string>();
        for (const s of snapshots) {
          const ch = s.nodes[nodeId]?.channels[chId];
          if (ch) {
            for (const subId of Object.keys(ch.subscribers)) subIds.add(subId);
          }
        }
        channelSubIds[chId] = subIds;
      }

      const channelHistory: Record<string, ChannelHistory> = {};
      for (const chId of channelIds) {
        const subHist: Record<string, SubscriberHistory> = {};
        for (const subId of channelSubIds[chId]!) {
          subHist[subId] = { msgThroughput: [], byteThroughput: [] };
        }
        channelHistory[chId] = { msgThroughput: [], byteThroughput: [], bufferDepths: [], subscriberHistory: subHist };
      }

      for (let i = 0; i < snapshots.length; i++) {
        const s = snapshots[i]!;
        const d = dts[i]!;
        const rate = d > 0 ? 1 / d : 0;
        const n = s.nodes[nodeId];

        if (!n) {
          msgThroughput.push(0);
          byteThroughput.push(0);
          for (const chId of channelIds) {
            channelHistory[chId]!.msgThroughput.push(0);
            channelHistory[chId]!.byteThroughput.push(0);
            channelHistory[chId]!.bufferDepths.push(0);
            for (const subId of channelSubIds[chId]!) {
              channelHistory[chId]!.subscriberHistory[subId]!.msgThroughput.push(0);
              channelHistory[chId]!.subscriberHistory[subId]!.byteThroughput.push(0);
            }
          }
          continue;
        }

        let totalMsg = 0;
        let totalBytes = 0;
        for (const ch of Object.values(n.channels)) {
          totalMsg += ch.msg_count_delta;
          totalBytes += ch.byte_count_delta;
        }
        msgThroughput.push(totalMsg * rate);
        byteThroughput.push(totalBytes * rate);

        for (const chId of channelIds) {
          const ch = n.channels[chId];
          channelHistory[chId]!.msgThroughput.push((ch?.msg_count_delta ?? 0) * rate);
          channelHistory[chId]!.byteThroughput.push((ch?.byte_count_delta ?? 0) * rate);
          channelHistory[chId]!.bufferDepths.push(ch?.buffer_depth ?? 0);

          for (const subId of channelSubIds[chId]!) {
            const sub = ch?.subscribers[subId];
            channelHistory[chId]!.subscriberHistory[subId]!.msgThroughput.push((sub?.msg_count_delta ?? 0) * rate);
            channelHistory[chId]!.subscriberHistory[subId]!.byteThroughput.push((sub?.byte_count_delta ?? 0) * rate);
          }
        }
      }

      nodeHistory[nodeId] = { msgThroughput, byteThroughput, channelHistory };
    }

    setHistory({ current: snapshot, snapshots, snapshotRate, dt, nodeHistory });
  }, [snapshot]);

  return history;
}

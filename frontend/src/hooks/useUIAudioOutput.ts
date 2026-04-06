import { useEffect, useRef } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

export function useUIAudioOutput(nodeId: string | null, channel: string) {
  const { subscribe } = useUIChannel();
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  // Adaptive jitter buffer: starts at 50ms, grows on underruns, decays on stability
  const jitterRef = useRef<number>(0.05);
  const successCountRef = useRef<number>(0);

  useEffect(() => {
    if (!nodeId) return;

    const unsub = subscribe(nodeId, channel, (payload: unknown) => {
      if (payload instanceof ArrayBuffer) {
        if (!audioContextRef.current) {
          audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
          nextStartTimeRef.current = audioContextRef.current.currentTime;
        }

        const ctx = audioContextRef.current;
        // Assume 24000 Hz 1-channel Int16 PCM for now as default from SpeakerBrowser
        const sampleRate = 24000;

        // Convert Int16Array to Float32Array
        const int16Array = new Int16Array(payload);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
          float32Array[i] = int16Array[i] / 32768.0;
        }

        const audioBuffer = ctx.createBuffer(1, float32Array.length, sampleRate);
        audioBuffer.copyToChannel(float32Array, 0);

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);

        const currentTime = ctx.currentTime;
        let scheduledTime = nextStartTimeRef.current;

        if (scheduledTime < currentTime) {
          // Underrun — grow jitter buffer (capped at 250ms)
          jitterRef.current = Math.min(jitterRef.current * 1.5, 0.25);
          scheduledTime = currentTime + jitterRef.current;
          successCountRef.current = 0;
        } else {
          // Stable — slowly decay jitter buffer toward 50ms
          successCountRef.current++;
          if (successCountRef.current > 100) {
            jitterRef.current = Math.max(jitterRef.current * 0.8, 0.05);
            successCountRef.current = 0;
          }
        }

        // Schedule the audio chunk and advance the timeline
        source.start(scheduledTime);
        nextStartTimeRef.current = scheduledTime + audioBuffer.duration;
      }
    });

    return () => {
      unsub();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
    };
  }, [nodeId, channel, subscribe]);
}

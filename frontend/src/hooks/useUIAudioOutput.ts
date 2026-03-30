import { useEffect, useRef } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

export function useUIAudioOutput(nodeId: string | null, channel: string) {
  const { subscribe } = useUIChannel();
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);

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

        // If we underrun (meaning the buffer completely emptied out before 
        // the next chunk arrived), we need to add a substantial jitter buffer.
        // A 250ms buffer allows the stream to absorb network latency spikes
        // from the server TTS stream without creating micro-stutters and repeats.
        if (scheduledTime < currentTime) {
          scheduledTime = currentTime + 0.25; 
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

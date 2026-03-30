import { useEffect, useRef, useState, useCallback } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

export function useUIAudioInput(nodeId: string | null, channel: string) {
  const { sendUIBinary } = useUIChannel();
  const [isRecording, setIsRecording] = useState(false);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const startRecording = useCallback(async () => {
    if (!nodeId) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      mediaStreamRef.current = stream;

      const sampleRate = 16000; // Expected by MicBrowser
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate,
      });

      const ctx = audioContextRef.current;
      const source = ctx.createMediaStreamSource(stream);

      // Buffer sizes must be a power of 2, like 1024 or 2048. Smaller = less latency.
      const processor = ctx.createScriptProcessor(1024, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Convert Float32 to Int16
        const int16Buffer = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        sendUIBinary(nodeId, channel, int16Buffer.buffer);
      };

      source.connect(processor);
      processor.connect(ctx.destination); // Required for processor to start in some browsers

      setIsRecording(true);
    } catch (err) {
      console.error("Failed to access microphone:", err);
    }
  }, [nodeId, channel, sendUIBinary]);

  const stopRecording = useCallback(() => {
    if (processorRef.current && audioContextRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    setIsRecording(false);
  }, []);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return { isRecording, startRecording, stopRecording };
}

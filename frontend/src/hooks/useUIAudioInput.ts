import { useEffect, useRef, useState, useCallback } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";
import { getAudioInputProcessorUrl } from "@/lib/audioInputProcessor";

export function useUIAudioInput(nodeId: string | null, channel: string) {
  const { sendUIBinary } = useUIChannel();
  const [isRecording, setIsRecording] = useState(false);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

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
      await ctx.audioWorklet.addModule(getAudioInputProcessorUrl());

      const workletNode = new AudioWorkletNode(ctx, "audio-input-processor");
      workletNodeRef.current = workletNode;

      workletNode.port.onmessage = (e: MessageEvent) => {
        sendUIBinary(nodeId, channel, e.data as ArrayBuffer);
      };

      const source = ctx.createMediaStreamSource(stream);
      source.connect(workletNode);
      // AudioWorklet does not need to connect to destination

      setIsRecording(true);
    } catch (err) {
      console.error("Failed to access microphone:", err);
    }
  }, [nodeId, channel, sendUIBinary]);

  const stopRecording = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
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

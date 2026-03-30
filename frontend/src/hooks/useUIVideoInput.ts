import { useEffect, useRef, useState, useCallback } from "react";
import { useUIChannel } from "@/contexts/UIChannelContext";

export function useUIVideoInput(nodeId: string | null, channel: string) {
  const { sendUIBinary } = useUIChannel();
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number>(0);

  const startRecording = useCallback(async () => {
    if (!nodeId || !videoRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480, frameRate: { ideal: 15 } }, 
          audio: false 
      });
      mediaStreamRef.current = stream;
      
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      if (!ctx) return;

      const sendFrame = () => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          canvas.toBlob((blob) => {
            if (blob) {
              blob.arrayBuffer().then(buffer => {
                sendUIBinary(nodeId, channel, buffer);
              });
            }
          }, 'image/jpeg', 0.5); // Adjust quality as needed (0.5 for lower bandwidth)
        }
      };

      // Send at ~10 FPS (100ms) to conserve standard websocket bandwidth
      intervalRef.current = window.setInterval(sendFrame, 100);
      setIsRecording(true);
    } catch (err) {
      console.error("Failed to access camera:", err);
    }
  }, [nodeId, channel, sendUIBinary]);

  const stopRecording = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = 0;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
        videoRef.current.srcObject = null;
    }
    setIsRecording(false);
  }, []);

  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, [stopRecording]);

  return { isRecording, startRecording, stopRecording, videoRef };
}

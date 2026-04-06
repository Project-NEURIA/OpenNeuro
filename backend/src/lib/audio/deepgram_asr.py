"""DeepgramASR — streaming speech-to-text with built-in VAD via Deepgram's WebSocket API.

Replaces the separate VAD + ASR pipeline with a single component.
Send AudioFrames in, get TextFrames out.
"""

from __future__ import annotations

import json
import os
import threading
from typing import NamedTuple

from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import ThreadedComponent, Tag
from src.core.frames import AudioDataFormat, AudioFrame, TextFrame


class DeepgramASRConfig(BaseModel):
    api_key_env_var: str = "DEEPGRAM_API_KEY"
    model: str = "nova-3"
    language: str = "en"
    sample_rate: int = 48000
    channels: int = 1
    interim_results: bool = True
    utterance_end_ms: int = 1000
    vad_events: bool = True
    smart_format: bool = True
    diarize: bool = True


class DeepgramASRInputs(NamedTuple):
    audio: Receiver[AudioFrame]


class DeepgramASROutputs(NamedTuple):
    text: Sender[TextFrame]


class DeepgramASR(ThreadedComponent[DeepgramASRInputs, DeepgramASROutputs]):
    tags = Tag(io={"conduit"}, functionality={"audio"})
    description = (
        "Streaming speech-to-text via **Deepgram** WebSocket API with built-in VAD. "
        "Replaces separate VAD + ASR components. Send AudioFrames in, get TextFrames out."
    )

    def __init__(self, config: DeepgramASRConfig = DeepgramASRConfig()) -> None:
        super().__init__()
        self.config = config

    def run(self, inputs: DeepgramASRInputs, outputs: DeepgramASROutputs) -> None:
        import websockets.sync.client as ws_client

        api_key = os.getenv(self.config.api_key_env_var, "")
        if not api_key:
            print(f"[DeepgramASR] {self.config.api_key_env_var} not set — stopping")
            return

        c = self.config
        params = (
            f"model={c.model}"
            f"&language={c.language}"
            f"&sample_rate={c.sample_rate}"
            f"&channels={c.channels}"
            f"&encoding=linear16"
            f"&interim_results={'true' if c.interim_results else 'false'}"
            f"&utterance_end_ms={c.utterance_end_ms}"
            f"&vad_events={'true' if c.vad_events else 'false'}"
            f"&smart_format={'true' if c.smart_format else 'false'}"
            f"&diarize={'true' if c.diarize else 'false'}"
        )
        url = f"wss://api.deepgram.com/v1/listen?{params}"

        print(f"[DeepgramASR] Connecting to Deepgram ({c.model})")

        try:
            ws = ws_client.connect(
                url,
                additional_headers={"Authorization": f"Token {api_key}"},
            )
        except Exception as e:
            print(f"[DeepgramASR] Connection failed: {e}")
            print(f"[DeepgramASR] URL: {url}")
            return

        print("[DeepgramASR] Connected")

        # Receiver thread: read transcription results from Deepgram
        def recv_loop() -> None:
            try:
                for msg in ws:
                    if self.stop_event.is_set():
                        break
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        msg_type = data.get("type", "")

                        if msg_type == "Results":
                            channel = data.get("channel", {})
                            alternatives = channel.get("alternatives", [])
                            if alternatives:
                                alt = alternatives[0]
                                transcript = alt.get("transcript", "").strip()
                                is_final = data.get("is_final", False)
                                if transcript and is_final:
                                    # Extract speaker from first word's diarization
                                    words = alt.get("words", [])
                                    speaker = (
                                        words[0].get("speaker", None) if words else None
                                    )
                                    if speaker is not None:
                                        text = f"[Speaker {speaker}] {transcript}"
                                    else:
                                        text = transcript
                                    print(f"[DeepgramASR] {text}")
                                    outputs.text.send(TextFrame.new(text=text))

                        elif msg_type == "UtteranceEnd":
                            pass  # Could be used for turn-taking signals

            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"[DeepgramASR] Receive error: {e}")

        recv_thread = threading.Thread(target=recv_loop, daemon=True)
        recv_thread.start()

        # Send audio frames to Deepgram
        try:
            for frame in inputs.audio:
                if frame is None or self.stop_event.is_set():
                    break
                pcm = frame.get(
                    sample_rate=c.sample_rate,
                    num_channels=c.channels,
                    data_format=AudioDataFormat.PCM16,
                )
                try:
                    ws.send(pcm)
                except Exception:
                    break
        finally:
            # Send close message to Deepgram
            try:
                ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                ws.close()
            except Exception:
                pass
            recv_thread.join(timeout=2.0)
            print("[DeepgramASR] Stopped")

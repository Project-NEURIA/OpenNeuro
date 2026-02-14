from __future__ import annotations

import base64
import json
import os
import threading
import time

import numpy as np
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from openneuro.config import get_config_value
from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import AudioFrame

class STS(Component[Channel[AudioFrame]]):
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        user_name: str | None = None,
        chatbot_name: str | None = None,
    ) -> None:
        super().__init__()
        self._output = Channel[AudioFrame]()
        self._ws: Connection | None = None
        self._last_response_item: str | None = None
        self._response_start_time: float | None = None
        self._is_speaking = False
        self._lock = threading.Lock()
        
        # Load from config if not provided
        self._system_prompt = system_prompt or get_config_value("system_prompt", "You are Dan.")
        self._user_name = user_name or get_config_value("user_name", "User")
        self._chatbot_name = chatbot_name or get_config_value("chatbot_name", "Assistant")

    def get_output_channels(self) -> tuple[Channel[AudioFrame]]:
        return (self._output,)

    def set_input_channels(self, t1: Channel[AudioFrame]) -> None:
        self._input_channel = t1

    def stop(self) -> None:
        # Close WebSocket to unblock the recv loop
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        super().stop()

    def run(self) -> None:
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        headers = {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "OpenAI-Beta": "realtime=v1",
        }

        with connect(url, additional_headers=headers) as ws:
            self._ws = ws
            ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad"},
                    "tools": [],
                    "instructions": f"You are {self._chatbot_name}. {self._system_prompt}",
                },
            }))
            
            # ws.send(json.dumps({
            #     "type": "conversation.item.create",
            #     "item": {
            #         "type": "message",
            #         "role": "system",
            #         "content": [
            #             {
            #                 "type": "input_text",
            #                 "text": f"You are {self._chatbot_name}. {self._system_prompt}"
            #             }
            #         ]
            #     }
            # }))

            threading.Thread(target=self._send_loop, args=(ws,), daemon=True).start()

            for msg in ws:
                if self.stop_event.is_set():
                    break
                event = json.loads(msg)
                
                # Handle speech started detection for interrupts
                if event["type"] == "input_audio_buffer.speech_started":
                    self._handle_speech_started(ws)
                
                elif event["type"] == "response.audio.delta":
                    pcm = base64.b64decode(event["delta"])
                    # Create AudioFrame from received PCM data
                    audio_frame = AudioFrame(
                        frame_type_string="sts_output",
                        pcm16_data=pcm,
                        sample_rate=24000,  # OpenAI realtime API outputs 24kHz
                        channels=1
                    )
                    
                    with self._lock:
                        self._is_speaking = True
                        if self._response_start_time is None:
                            self._response_start_time = time.time()
                    
                    self._output.send(audio_frame)
                
                # Track response items for interrupt handling
                elif event["type"] == "response.item.created":
                    with self._lock:
                        self._last_response_item = event.get("item", {}).get("id")
                
                # Reset speaking state when response is done
                elif event["type"] == "response.done":
                    with self._lock:
                        self._is_speaking = False
                        self._response_start_time = None
                        self._last_response_item = None
                        
                else:
                    print(f"{event}")

    def _send_loop(self, ws: Connection) -> None:
        from websockets.exceptions import ConnectionClosed

        for audio_frame in self._input_channel.stream(self.stop_event):
            if audio_frame is None:
                break
            
            # Resample to 24kHz if needed for OpenAI API
            if audio_frame.sample_rate != 24000:
                audio_frame = audio_frame.resample(24000)
            
            # Convert AudioFrame to numpy array
            pcm_data = np.frombuffer(audio_frame.pcm16_data, dtype=np.int16)
            
            # Convert to base64 for WebSocket transmission
            b64 = base64.b64encode(pcm_data.tobytes()).decode()
            try:
                ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64,
                }))
            except ConnectionClosed:
                break

    def _handle_speech_started(self, ws: Connection) -> None:
        """Handle interrupt when user speech is detected during AI response."""
        with self._lock:
            if self._is_speaking and self._last_response_item:
                print(f"Interrupting response with item ID: {self._last_response_item}")
                
                # Calculate elapsed time for truncation
                elapsed_time = 0
                if self._response_start_time is not None:
                    elapsed_time = int((time.time() - self._response_start_time) * 1000)  # Convert to ms
                
                # Send truncation event to OpenAI
                truncate_event = {
                    "type": "conversation.item.truncate",
                    "item_id": self._last_response_item,
                    "content_index": 0,
                    "audio_end_ms": elapsed_time
                }
                
                try:
                    ws.send(json.dumps(truncate_event))
                    print(f"Sent truncation event for {elapsed_time}ms of audio")
                except Exception as e:
                    print(f"Error sending truncation: {e}")
                
                # Reset state
                self._is_speaking = False
                self._response_start_time = None
                self._last_response_item = None

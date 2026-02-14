from __future__ import annotations

import threading
import time

from openneuro.config import get_config_value
from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import ConversationalFrame, InterruptFrame, TextFrame


class AgentState(Component[Channel[TextFrame], Channel[TextFrame]]):
    """Agent state component that manages conversation history.
    
    This component maintains chat history and builds context for LLM APIs.
    It has 2 input streams:
    - Slot 0: ASR transcriptions (user input)
    - Slot 1: LLM/TTS text chunks (bot output for history)
    
    Outputs ConversationalFrame with context/messages for LLM consumption.
    """

    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        user_name: str | None = None,
        chatbot_name: str | None = None,
    ) -> None:
        super().__init__()
        self._output = Channel[ConversationalFrame]()
        
        # Load from config if not provided
        self._system_prompt = system_prompt or get_config_value("system_prompt", "You are a helpful AI assistant.")
        self._user_name = user_name or get_config_value("user_name", "User")
        self._chatbot_name = chatbot_name or get_config_value("chatbot_name", "Assistant")
        
        # Conversation history as list of (speaker, text) tuples
        self._history: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        
        # Input channels
        self._asr_input: Channel[TextFrame] | None = None
        self._feedback_input: Channel[TextFrame] | None = None
        
        # Track current bot response being built
        self._current_bot_response = ""

    def get_output_channels(self) -> tuple[Channel[ConversationalFrame]]:
        return (self._output,)

    @classmethod
    def get_input_names(cls) -> list[str]:
        """Returns descriptive names for the two input slots."""
        return ["asr", "feedback"]

    def set_input_channels(self, asr_input: Channel[TextFrame], feedback_input: Channel[TextFrame]) -> None:
        self._asr_input = asr_input
        self._feedback_input = feedback_input

    def _build_context(self) -> str:
        """Build single prompt string for completions-style APIs."""
        with self._lock:
            lines = [self._system_prompt, "***"]
            for name, text in self._history:
                lines.append(f"{name}: {text}")
            lines.append(f"{self._chatbot_name}:")
            return "\n".join(lines)

    def _build_messages(self) -> list[dict[str, str]]:
        """Build conversation list for chat completions-style APIs."""
        with self._lock:
            messages = [{"role": "system", "content": self._system_prompt}]
            for name, text in self._history:
                if name == self._user_name:
                    messages.append({"role": "user", "content": text})
                elif name == self._chatbot_name:
                    messages.append({"role": "assistant", "content": text})
            return messages

    def _process_asr_input(self) -> None:
        """Process ASR transcriptions from slot 0."""
        print("[AgentState] Starting ASR input processing")
        
        if self._asr_input is None:
            print("[AgentState] ASR input not connected")
            return
            
        for text_frame in self._asr_input.stream(self.stop_event):
            if text_frame is None:
                break
                
            # Handle interrupt frames
            if isinstance(text_frame, InterruptFrame):
                print("[AgentState] Received interrupt")
                # Pass through interrupt to output
                self._output.send(text_frame)
                continue
            
            # Process user transcription
            if text_frame.frame_type_string == "asr_transcription":
                text = text_frame.text.strip()
                if not text:
                    continue
                    
                with self._lock:
                    self._history.append((self._user_name, text))
                
                print(f"[AgentState] User: {text}")
                
                # Output context for LLM as ConversationalFrame
                context = self._build_context()
                messages = self._build_messages()
                
                state_frame = ConversationalFrame(
                    frame_type_string="agent_state",
                    text=context,
                    messages=messages,
                    language="en",
                    pts=text_frame.pts
                )
                self._output.send(state_frame)

    def _process_feedback_input(self) -> None:
        """Process LLM/TTS feedback from slot 1."""
        print("[AgentState] Starting feedback input processing")
        
        if self._feedback_input is None:
            print("[AgentState] Feedback input not connected")
            return
            
        for text_frame in self._feedback_input.stream(self.stop_event):
            if text_frame is None:
                break
            
            # Process bot text chunks
            if text_frame.frame_type_string in ("llm_chunk", "tts_text"):
                chunk = text_frame.text
                if not chunk:
                    continue
                
                with self._lock:
                    # Check if we're continuing the last assistant message
                    if self._history and self._history[-1][0] == self._chatbot_name:
                        name, text = self._history[-1]
                        self._history[-1] = (name, text + chunk)
                    else:
                        self._history.append((self._chatbot_name, chunk))
                
                print(f"[AgentState] Bot chunk: {chunk[:50]}...")

    def run(self) -> None:
        """Main processing loop - runs both input processors in parallel threads."""
        print("[AgentState] Starting Agent State management")
        
        # Wait for input channels to be set
        while (self._asr_input is None or self._feedback_input is None) and not self.stop_event.is_set():
            print("[AgentState] Waiting for input channels...")
            time.sleep(0.1)
        
        if self.stop_event.is_set():
            return
        
        # Start both input processors in separate threads
        asr_thread = threading.Thread(target=self._process_asr_input, daemon=True)
        feedback_thread = threading.Thread(target=self._process_feedback_input, daemon=True)
        
        asr_thread.start()
        feedback_thread.start()
        
        # Wait for threads to complete
        asr_thread.join()
        feedback_thread.join()
        
        print("[AgentState] Agent State management stopped")

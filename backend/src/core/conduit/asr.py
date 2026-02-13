from __future__ import annotations

import os
import tempfile
import wave
from typing import TypedDict

import numpy as np
import requests

from src.core.component import Component
from src.core.channel import Channel


class ASROutputs(TypedDict):
    text: Channel[str]


class ASR(Component[[Channel[bytes]], ASROutputs]):
    def __init__(self) -> None:
        super().__init__()
        self._output_text: Channel[str] = Channel(name="text")

    def output_channels(self) -> ASROutputs:
        return {"text": self._output_text}

    def run(self, audio: Channel[bytes]) -> None:
        for pcm48 in audio.stream(self):
            if pcm48 is None:
                break
            pcm = np.frombuffer(pcm48, dtype=np.int16)
            mono16k = pcm[::3]

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            try:
                tmp.close()

                wf = wave.open(tmp_path, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(mono16k.tobytes())
                wf.close()

                headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
                with open(tmp_path, "rb") as fp:
                    files = {"file": ("audio.wav", fp, "audio/wav")}
                    data = {"model": "whisper-large-v3-turbo"}
                    r = requests.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers, files=files, data=data, timeout=60,
                    )

                text = r.json().get("text", "")
                if text:
                    self._output_text.send(text)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

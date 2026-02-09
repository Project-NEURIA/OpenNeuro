from __future__ import annotations

import os
import tempfile
import wave

import numpy as np
import requests

from ..node import Node
from ..topic import Topic


class ASR(Node[str]):
    def __init__(self, input_topic: Topic[bytes]) -> None:
        self._input_topic = input_topic
        super().__init__(Topic[str]())

    def set_input_topics(self, *topics: Topic) -> None:
        self._input_topic = topics[0]

    def run(self) -> None:
        for pcm48 in self._input_topic.stream(self.stop_event):
            if self._stopped:
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
                    self.topic.send(text)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

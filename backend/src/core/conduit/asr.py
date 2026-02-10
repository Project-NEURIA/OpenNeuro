from __future__ import annotations

import os
import tempfile
import wave
from typing import Never

import numpy as np
import requests

from ..component import Component
from ..topic import NOTOPIC, Topic


class ASR(Component[bytes, str]):
    def __init__(self) -> None:
        super().__init__()
        self._output = Topic[str]()

    def get_output_topics(self) -> tuple[Topic[str], Topic[Never], Topic[Never], Topic[Never]]:
        return (self._output, NOTOPIC, NOTOPIC, NOTOPIC)

    def set_input_topics(self, t1: Topic[bytes], t2: Topic[Never] = NOTOPIC, t3: Topic[Never] = NOTOPIC, t4: Topic[Never] = NOTOPIC) -> None:
        self._input_topic = t1

    def run(self) -> None:
        for pcm48 in self._input_topic.stream(self.stop_event):
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
                    self._output.send(text)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

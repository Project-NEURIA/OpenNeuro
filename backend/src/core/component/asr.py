from __future__ import annotations

import os
import tempfile
import wave

import numpy as np
import requests

from ..base import Base, Streamable
from .models import AudioSegment, Transcription


class ASR(Base[Transcription]):
    def __init__(
        self,
        source: Streamable[AudioSegment],
        api_key: str,
        *,
        url: str = "https://api.groq.com/openai/v1/audio/transcriptions",
        model: str = "whisper-large-v3-turbo",
    ) -> None:
        super().__init__()
        self._source = source
        self._api_key = api_key
        self._url = url
        self._model = model

    def run(self) -> None:
        for segment in self._source.stream():
            pcm48 = segment.pcm48_stereo

            pcm = np.frombuffer(pcm48, dtype=np.int16).reshape(-1, 2).mean(axis=1).astype(np.int16)
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

                headers = {"Authorization": f"Bearer {self._api_key}"}
                with open(tmp_path, "rb") as fp:
                    files = {"file": ("audio.wav", fp, "audio/wav")}
                    data = {"model": self._model}
                    r = requests.post(self._url, headers=headers, files=files, data=data, timeout=60)

                text = r.json().get("text", "")
                if text:
                    self.send(Transcription(text=text))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

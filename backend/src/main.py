from __future__ import annotations

import threading

from dotenv import load_dotenv

from .core.source import Microphone
from .core.sink import Speaker
from .core.conduit import VAD, ASR, LLM, TTS, STS


def main() -> None:
    load_dotenv(override=True)

    mic = Microphone()
    vad = VAD(mic.topic.stream())
    asr = ASR(vad.topic.stream())
    llm = LLM(asr.topic.stream())
    tts = TTS(llm.topic.stream())
    speaker = Speaker(tts.topic.stream())

    mic.start()
    vad.start()
    asr.start()
    llm.start()
    tts.start()
    speaker.start()

    threading.Event().wait()


def main2() -> None:
    load_dotenv(override=True)

    mic = Microphone()
    sts = STS(mic.topic.stream())
    speaker = Speaker(sts.topic.stream())

    mic.start()
    sts.start()
    speaker.start()

    threading.Event().wait()


if __name__ == "__main__":
    main2()

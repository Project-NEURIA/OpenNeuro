from __future__ import annotations

__all__ = ["QwenTTS"]


def __getattr__(name: str):
    if name == "QwenTTS":
        from src.core.conduit.qwen_tts.component import QwenTTS

        return QwenTTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

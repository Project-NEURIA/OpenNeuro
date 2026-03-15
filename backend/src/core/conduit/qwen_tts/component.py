"""QwenTTS conduit — local text-to-speech via Qwen3 TTS."""

from __future__ import annotations

import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent, Tag
from src.core.utils import StreamFilter
from src.core.frames import AudioFrame, EOS, InterruptFrame, TextFrame

_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "qwen_tts_voices"
_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


class QwenTTSConfig(BaseModel):
    model_config = ConfigDict(json_schema_extra={"configOptions": {"voice_id": {}}})

    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    device: str = "auto"
    voice_id: str = ""
    ref_samples_dir: str = str(_ASSETS_DIR)
    language: str = "English"


class QwenTTSInputs(NamedTuple):
    text: Receiver[TextFrame | EOS]
    interrupt: Receiver[InterruptFrame] | None = None


class QwenTTSOutputs(NamedTuple):
    audio: Sender[AudioFrame]
    text: Sender[TextFrame]


class QwenTTS(PrimitiveComponent[QwenTTSInputs, QwenTTSOutputs]):
    """Text-to-Speech using local Qwen3 TTS model."""

    _tags = Tag(io={"conduit"}, functionality={"audio"}, gpu={"cpu", "nvidia", "apple"})

    def __init__(self, config: QwenTTSConfig) -> None:
        super().__init__()
        self.config = config
        self._stream_filter = StreamFilter()
        self._generation = 0
        self._gen_lock = threading.Lock()
        self._task_queue: Queue[tuple[int, str]] = Queue()
        self._tts: Any = None
        self._voice_paths: dict[str, Path] = {}

    @classmethod
    def get_config_options(
        cls, field: str, values: dict[str, Any] | None = None
    ) -> list[dict[str, str]] | None:
        if field != "config.voice_id":
            return None
        ref_dir = Path((values or {}).get("ref_samples_dir", str(_ASSETS_DIR)))
        options: list[dict[str, str]] = []
        if ref_dir.is_dir():
            for p in sorted(ref_dir.iterdir()):
                if p.suffix.lower() in _AUDIO_EXTENSIONS:
                    txt = ref_dir / f"{p.stem}.txt"
                    if txt.is_file():
                        options.append({"value": p.stem, "label": p.stem})
        return options or None

    def _load_model(self) -> None:
        from src.core.conduit.qwen_tts.model import SimpleStreamingTTS

        self._tts = SimpleStreamingTTS.load(self.config.model_id, self.config.device)

    def _register_voices(self) -> None:
        ref_dir = Path(self.config.ref_samples_dir)
        if not ref_dir.is_dir():
            print(f"[QwenTTS] ref_samples_dir not found: {ref_dir}")
            return
        for p in sorted(ref_dir.iterdir()):
            if p.suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            txt_path = ref_dir / f"{p.stem}.txt"
            if not txt_path.is_file():
                continue
            transcript = txt_path.read_text(encoding="utf-8").strip()
            if not transcript:
                continue
            try:
                path = self._tts.register_voice(p.stem, str(p), transcript)
                self._voice_paths[p.stem] = path
                print(f"[QwenTTS] Registered voice: {p.stem}")
            except Exception as e:
                print(f"[QwenTTS] Failed to register {p.stem}: {e}")

    def _resolve_voice(self) -> Path:
        vid = self.config.voice_id
        if vid and vid in self._voice_paths:
            return self._voice_paths[vid]
        if self._voice_paths:
            return next(iter(self._voice_paths.values()))
        raise RuntimeError("[QwenTTS] No voices registered")

    def _worker(self, outputs: QwenTTSOutputs) -> None:
        print("[QwenTTS] Worker thread started")
        voice_path = self._resolve_voice()

        while not self.stop_event.is_set():
            try:
                gen, text = self._task_queue.get(timeout=0.1)
                with self._gen_lock:
                    if gen != self._generation:
                        continue

                cancel = threading.Event()
                for audio_chunk, meta in self._tts.generate_streaming(
                    text, voice_path, self.config.language
                ):
                    if cancel.is_set():
                        break
                    with self._gen_lock:
                        if gen != self._generation:
                            cancel.set()
                            break

                    if meta.get("is_final") or len(audio_chunk) == 0:
                        continue

                    outputs.audio.send(
                        AudioFrame.new(
                            data=audio_chunk,
                            sample_rate=self._tts.sample_rate,
                            channels=1,
                        )
                    )

                with self._gen_lock:
                    if gen == self._generation:
                        outputs.text.send(TextFrame.new(text=text))
            except Empty:
                continue
            except Exception as e:
                print(f"[QwenTTS] Generation error: {e}")

    def run(self, inputs: QwenTTSInputs, outputs: QwenTTSOutputs) -> None:
        print("[QwenTTS] Starting QwenTTS")

        self._load_model()
        self._register_voices()

        worker_thread = threading.Thread(
            target=self._worker, args=(outputs,), daemon=True
        )
        worker_thread.start()

        if inputs.interrupt is not None:
            interrupt_recv = inputs.interrupt

            def handle_interrupts() -> None:
                for frame in interrupt_recv(self):
                    if frame is None:
                        break
                    print(f"[QwenTTS] Interrupt received: {frame.reason}")
                    with self._gen_lock:
                        self._generation += 1
                    self._stream_filter = StreamFilter()
                    while not self._task_queue.empty():
                        try:
                            self._task_queue.get_nowait()
                        except Empty:
                            break

            threading.Thread(target=handle_interrupts, daemon=True).start()

        for frame in inputs.text(self):
            if frame is None:
                break
            if frame is EOS.END:
                out = self._stream_filter.feed("", force=True)
            else:
                out = self._stream_filter.feed(frame.text)
            if out and out.strip():
                with self._gen_lock:
                    gen = self._generation
                self._task_queue.put((gen, out))

        worker_thread.join(timeout=1)
        print("[QwenTTS] QwenTTS stopped")

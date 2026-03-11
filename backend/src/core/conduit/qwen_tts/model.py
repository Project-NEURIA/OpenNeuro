"""Qwen3 TTS streaming inference — adapted from qwen3streaming.py with cross-platform support."""

from __future__ import annotations

import functools
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Union

import librosa
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
)
from src.core.conduit.qwen_tts.tts_model.modeling_qwen3_tts_tokenizer_v2 import (  # noqa: F401
    Qwen3TTSTokenizerV2Model as Qwen3TTSTokenizerV2Model,
)
from src.core.conduit.qwen_tts.tts_model.processing_qwen3_tts import (
    Qwen3TTSProcessor,
)

AudioLike = Union[str, tuple[np.ndarray, int]]


@dataclass
class VoiceClonePromptItem:
    ref_code: torch.Tensor | None
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


@dataclass
class StreamingConfig:
    min_initial_frames: int = 4
    yield_every_n_frames: int = 1


_streaming_config = StreamingConfig()


class Qwen3TTSModel:
    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor: Any) -> None:
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        self.generate_defaults: dict[str, Any] = (
            getattr(model, "generate_config", {}) or {}
        )

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> Qwen3TTSModel:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"Expected Qwen3TTSForConditionalGeneration, got {type(model)}"
            )
        processor = AutoProcessor.from_pretrained(path, fix_mistral_regex=True)
        return cls(model, processor)

    @staticmethod
    def _as_list(x: Any) -> list[Any]:
        return x if isinstance(x, list) else [x]

    @staticmethod
    def _assistant_text(t: str) -> str:
        return f"<|im_start|>assistant\n{t}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _tokenize(self, texts: list[str]) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for t in texts:
            enc = self.processor(text=t, return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(self.device)
            out.append(ids.unsqueeze(0) if ids.dim() == 1 else ids)
        return out

    @staticmethod
    def _load_audio(a: AudioLike) -> tuple[np.ndarray, int]:
        if isinstance(a, str):
            wav, sr = librosa.load(a, sr=None, mono=True)
            return wav.astype(np.float32), int(sr)
        wav, sr = a
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        return wav.astype(np.float32), int(sr)

    @staticmethod
    def _items_to_prompt(items: list[VoiceClonePromptItem]) -> dict[str, Any]:
        return {
            "ref_code": [it.ref_code for it in items],
            "ref_spk_embedding": [it.ref_spk_embedding for it in items],
            "x_vector_only_mode": [it.x_vector_only_mode for it in items],
            "icl_mode": [it.icl_mode for it in items],
        }

    def _gen_defaults(self, **kwargs: Any) -> dict[str, Any]:
        hard = {
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "repetition_penalty": 1.05,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
            "max_new_tokens": 2048,
        }
        out = dict(kwargs)
        for k, v in hard.items():
            if k in out and out[k] is not None:
                continue
            if k in self.generate_defaults:
                out[k] = self.generate_defaults[k]
            else:
                out[k] = v
        return out

    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: AudioLike | list[AudioLike],
        ref_text: str | list[str | None] | None = None,
        x_vector_only_mode: bool | list[bool] = False,
    ) -> list[VoiceClonePromptItem]:
        if getattr(self.model, "tts_model_type", None) != "base":
            raise ValueError("create_voice_clone_prompt requires base model.")

        audios = self._as_list(ref_audio)
        texts = ref_text if isinstance(ref_text, list) else [ref_text] * len(audios)
        xvecs = (
            x_vector_only_mode
            if isinstance(x_vector_only_mode, list)
            else [x_vector_only_mode] * len(audios)
        )

        if not (len(audios) == len(texts) == len(xvecs)):
            raise ValueError("Batch size mismatch.")

        norm = [self._load_audio(a) for a in audios]
        wavs = [w for (w, _) in norm]
        srs = [sr for (_, sr) in norm]

        tokenizer = self.model.speech_tokenizer
        assert tokenizer is not None
        if len(set(srs)) == 1:
            enc = tokenizer.encode(wavs, sr=srs[0])
            codes = enc.audio_codes
        else:
            codes = [tokenizer.encode(w, sr=sr).audio_codes[0] for (w, sr) in norm]

        target_sr = int(self.model.speaker_encoder_sample_rate)
        items: list[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, t, xv) in enumerate(zip(norm, codes, texts, xvecs)):
            if (not xv) and (t is None or t == ""):
                raise ValueError(
                    f"ref_text required when x_vector_only_mode=False. index={i}"
                )
            wav2 = (
                wav
                if sr == target_sr
                else librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr)
            )
            spk = self.model.extract_speaker_embedding(audio=wav2, sr=target_sr)
            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xv else code,
                    ref_spk_embedding=spk,
                    x_vector_only_mode=bool(xv),
                    icl_mode=bool(not xv),
                    ref_text=t,
                )
            )
        return items

    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: str | list[str],
        language: str = "Auto",
        voice_clone_prompt: Any = None,
        **kwargs: Any,
    ) -> Any:
        if voice_clone_prompt is None:
            raise ValueError("voice_clone_prompt is required.")

        texts = self._as_list(text)
        langs = language if isinstance(language, list) else [language] * len(texts)
        if len(langs) == 1 and len(texts) > 1:
            langs = langs * len(texts)

        input_ids = self._tokenize([self._assistant_text(t) for t in texts])
        gen_kwargs = self._gen_defaults(**kwargs)

        ref_ids: list[torch.Tensor | None] | None = None
        if isinstance(voice_clone_prompt, list):
            prompt_items = voice_clone_prompt
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(
                    f"Batch mismatch: prompt={len(prompt_items)} text={len(texts)}"
                )
            ref_ids = []
            for it in prompt_items:
                if (
                    it is None
                    or it.ref_text is None
                    or it.ref_text == ""
                    or (not it.icl_mode)
                ):
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize([self._build_ref_text(it.ref_text)])[0]
                    ref_ids.append(ref_tok)
            voice_clone_prompt = self._items_to_prompt(prompt_items)

        return self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,  # type: ignore[arg-type]
            voice_clone_prompt=voice_clone_prompt,
            languages=langs,
            non_streaming_mode=False,
            **gen_kwargs,
        )


class SimpleStreamingTTS:
    """Streaming TTS using Qwen3 with cross-platform support."""

    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        processor: Any,
        device: torch.device,
    ) -> None:
        self._model = Qwen3TTSModel(model, processor)
        self._talker = model.talker
        self._decoder = model.speech_tokenizer.model.decoder  # type: ignore[attr-defined]
        self.sample_rate: int = model.speech_tokenizer.get_output_sample_rate()  # type: ignore[attr-defined]
        self._device = device

        # CUDA async decode stream (None on MPS/CPU)
        self._decode_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if device.type == "cuda" else None
        )

        self._voices_dir = Path(__file__).resolve().parent / "voices"
        self._voices_dir.mkdir(exist_ok=True)

    @classmethod
    def load(cls, model_id: str, device_cfg: str = "auto") -> SimpleStreamingTTS:
        """Load the Qwen3 TTS model with cross-platform device selection."""
        from src.core.utils import auto_device, auto_dtype

        device = auto_device(device_cfg)
        dtype = auto_dtype(device)

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]

        attn_impl = "flash_attention_2" if device.type == "cuda" else "eager"

        print(f"[QwenTTS] Loading model {model_id} on {device} ({dtype})...")
        wrapped = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=str(device),
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        tts = cls(wrapped.model, wrapped.processor, device)
        print(f"[QwenTTS] Loaded. Sample rate = {tts.sample_rate} Hz")
        return tts

    def register_voice(
        self,
        name: str,
        audio: str | tuple[np.ndarray, int],
        transcript: str,
        x_vector_only: bool = False,
    ) -> Path:
        print(f"[QwenTTS] Registering voice: {name}")
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=audio,
            ref_text=transcript,
            x_vector_only_mode=x_vector_only,
        )
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = self._voices_dir / f"{safe_name}.pt"
        torch.save(prompt, path)
        print(f"[QwenTTS] Saved voice prompt to {path}")
        return path

    def _decode_audio(self, codes: list[torch.Tensor]) -> np.ndarray:
        all_codes = torch.stack(codes, dim=-1)
        if self._decode_stream is not None:
            with torch.cuda.stream(self._decode_stream):
                with torch.no_grad():
                    audio = self._decoder(all_codes)
            self._decode_stream.synchronize()
        else:
            with torch.no_grad():
                audio = self._decoder(all_codes)
        return audio.squeeze().float().cpu().numpy()

    def generate_streaming(
        self,
        text: str,
        voice_prompt_path: str | Path,
        language: str = "English",
    ) -> Generator[tuple[np.ndarray, dict[str, Any]], None, None]:
        prompt = torch.load(voice_prompt_path, weights_only=False)
        code_queue: queue.Queue[torch.Tensor] = queue.Queue()
        generation_done = threading.Event()
        generation_error: list[Exception | None] = [None]

        original_forward = self._talker.forward

        @functools.wraps(original_forward)
        def hooked_forward(*args: Any, **kwargs: Any) -> Any:
            result = original_forward(*args, **kwargs)
            if hasattr(result, "hidden_states") and result.hidden_states is not None:
                hidden_states = result.hidden_states
                if isinstance(hidden_states, tuple) and len(hidden_states) == 2:
                    codec_ids = hidden_states[1]
                    if codec_ids is not None:
                        code_queue.put(codec_ids.detach().clone())
            return result

        def run_generation() -> None:
            try:
                self._talker.forward = hooked_forward
                self._model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                )
            except Exception as e:
                generation_error[0] = e
            finally:
                self._talker.forward = original_forward
                generation_done.set()

        threading.Thread(target=run_generation, daemon=True).start()

        codes: list[torch.Tensor] = []
        prev_audio_len = 0
        frame_idx = 0
        frames_since_yield = 0
        cfg = _streaming_config

        while True:
            try:
                code = code_queue.get(timeout=0.005)
                codes.append(code)
                frame_idx += 1
                frames_since_yield += 1

                should_yield = (
                    frame_idx >= cfg.min_initial_frames
                    and frames_since_yield >= cfg.yield_every_n_frames
                )
                if not should_yield:
                    continue

                audio = self._decode_audio(codes)
                new_audio = audio[prev_audio_len:]
                prev_audio_len = len(audio)

                yield new_audio, {"frame_idx": frame_idx}
                frames_since_yield = 0

            except queue.Empty:
                if generation_done.is_set():
                    break

        if generation_error[0]:
            raise generation_error[0]

        yield np.array([], dtype=np.float32), {"is_final": True}

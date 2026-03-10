#!/usr/bin/env python3
import os
import torch
import numpy as np
import time
import threading
import queue
import functools
from pathlib import Path
from typing import Generator, Tuple, Union
from dataclasses import dataclass
import librosa
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModel, AutoProcessor


from tts_model.modeling_qwen3_tts import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from tts_model.processing_qwen3_tts import Qwen3TTSProcessor
from tts_model.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model 

AudioLike = Union[str, Tuple[np.ndarray, int]]  # path OR (wav, sr)


@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[torch.Tensor]
    ref_spk_embedding: torch.Tensor
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class Qwen3TTSModel:
    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor: Any):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        self.generate_defaults = getattr(model, "generate_config", {}) or {}

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Qwen3TTSModel":
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(f"Expected Qwen3TTSForConditionalGeneration, got {type(model)}")

        processor = AutoProcessor.from_pretrained(path, fix_mistral_regex=True)
        return cls(model, processor)

    # --------- helpers ---------

    @staticmethod
    def _as_list(x):
        return x if isinstance(x, list) else [x]

    @staticmethod
    def _assistant_text(t: str) -> str:
        return f"<|im_start|>assistant\n{t}<|im_end|>\n<|im_start|>assistant\n"
    
    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _tokenize(self, texts: List[str]) -> List[torch.Tensor]:
        out = []
        for t in texts:
            enc = self.processor(text=t, return_tensors="pt", padding=True)
            ids = enc["input_ids"].to(self.device)
            out.append(ids.unsqueeze(0) if ids.dim() == 1 else ids)
        return out

    @staticmethod
    def _load_audio(a: AudioLike) -> Tuple[np.ndarray, int]:
        if isinstance(a, str):
            wav, sr = librosa.load(a, sr=None, mono=True)
            return wav.astype(np.float32), int(sr)
        wav, sr = a
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)
        return wav.astype(np.float32), int(sr)

    @staticmethod
    def _items_to_prompt(items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return {
            "ref_code": [it.ref_code for it in items],
            "ref_spk_embedding": [it.ref_spk_embedding for it in items],
            "x_vector_only_mode": [it.x_vector_only_mode for it in items],
            "icl_mode": [it.icl_mode for it in items],
        }

    def _gen_defaults(self, **kwargs) -> Dict[str, Any]:
        # minimal, but keeps your previous behavior of "use model defaults if present"
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

    # --------- required by SimpleStreamingTTS ---------

    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        if getattr(self.model, "tts_model_type", None) != "base":
            raise ValueError("create_voice_clone_prompt requires base model (tts_model_type=='base').")

        audios = self._as_list(ref_audio)
        texts = ref_text if isinstance(ref_text, list) else [ref_text] * len(audios)
        xvecs = x_vector_only_mode if isinstance(x_vector_only_mode, list) else [x_vector_only_mode] * len(audios)

        if not (len(audios) == len(texts) == len(xvecs)):
            raise ValueError("Batch size mismatch for ref_audio/ref_text/x_vector_only_mode.")

        norm = [self._load_audio(a) for a in audios]
        wavs = [w for (w, _) in norm]
        srs = [sr for (_, sr) in norm]

        # match upstream: batch encode if all sr same
        if len(set(srs)) == 1:
            enc = self.model.speech_tokenizer.encode(wavs, sr=srs[0])
            codes = enc.audio_codes
        else:
            codes = [self.model.speech_tokenizer.encode(w, sr=sr).audio_codes[0] for (w, sr) in norm]

        target_sr = int(self.model.speaker_encoder_sample_rate)
        items: List[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, t, xv) in enumerate(zip(norm, codes, texts, xvecs)):
            if (not xv) and (t is None or t == ""):
                raise ValueError(f"ref_text required when x_vector_only_mode=False (ICL). index={i}")

            wav2 = wav if sr == target_sr else librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr)
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
        text,
        language="Auto",
        voice_clone_prompt=None,
        non_streaming_mode=False,
        **kwargs,
    ):
        if voice_clone_prompt is None:
            raise ValueError("SimpleStreamingTTS path requires voice_clone_prompt.")

        texts = self._as_list(text)
        langs = language if isinstance(language, list) else [language] * len(texts)
        if len(langs) == 1 and len(texts) > 1:
            langs = langs * len(texts)

        input_ids = self._tokenize([self._assistant_text(t) for t in texts])
        gen_kwargs = self._gen_defaults(**kwargs)

        ref_ids = None

        # ✅ IMPORTANT: If prompt is a list (VoiceClonePromptItem), we can build ref_ids from ref_text
        if isinstance(voice_clone_prompt, list):
            prompt_items = voice_clone_prompt

            # batch-expand prompt if needed (same behavior as original)
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch mismatch: prompt={len(prompt_items)} text={len(texts)}")

            # build ref_ids for ICL mode
            ref_ids = []
            for it in prompt_items:
                if it is None or it.ref_text is None or it.ref_text == "" or (not it.icl_mode):
                    ref_ids.append(None)
                else:
                    # This matches the format the model expects (and later slices [:,3:-2])
                    ref_tok = self._tokenize([self._build_ref_text(it.ref_text)])[0]
                    ref_ids.append(ref_tok)

            voice_clone_prompt = self._items_to_prompt(prompt_items)

        return self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=langs,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

@dataclass
class StreamingConfig:
    min_initial_frames: int
    yield_every_n_frames: int

config = StreamingConfig(
    min_initial_frames=4,
    yield_every_n_frames=1,
)

# -----------------------------
# Model config: Hugging Face ID or local path (env TTS_MODEL_ID overrides)
# -----------------------------

MODEL_ID = os.environ.get("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
VOICES_DIR = Path(__file__).parent / "voices"
VOICES_DIR.mkdir(exist_ok=True)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SimpleStreamingTTS:
    def __init__(self):
        print(f"Loading base model from {MODEL_ID}...")
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        self.talker = self.model.model.talker
        
        self.decoder = self.model.model.speech_tokenizer.model.decoder
        self.sample_rate = self.model.model.speech_tokenizer.get_output_sample_rate()

        self.decode_stream = torch.cuda.Stream()

        print(f"Loaded. Sample rate = {self.sample_rate} Hz")

    # -----------------------------
    # Voice registration
    # -----------------------------

    def register_voice(
        self,
        name: str,
        audio: Union[str, Tuple[np.ndarray, int]],
        transcript: str,
        x_vector_only: bool = False,
    ) -> Path:
        print(f"Registering voice: {name}")

        prompt = self.model.create_voice_clone_prompt(
            ref_audio=audio,
            ref_text=transcript,
            x_vector_only_mode=x_vector_only,
        )

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = VOICES_DIR / f"{safe_name}.pt"
        torch.save(prompt, path)

        print(f"Saved voice prompt to {path}")
        return path

    # -----------------------------
    # True streaming generation
    # -----------------------------

    def generate_streaming(
        self,
        text: str,
        voice_prompt_path: Union[str, Path],
        language: str = "English",
    ) -> Generator[Tuple[np.ndarray, dict], None, None]:

        start_time = time.perf_counter()
        
        # --- Prompt load timing ---
        prompt = torch.load(voice_prompt_path, weights_only=False)
        prompt_load_time = time.perf_counter()

        code_queue = queue.Queue()
        generation_done = threading.Event()
        generation_error = [None]

        first_token_time = [None]
        first_audio_time = [None]
        token_times = []  # Track time of each token for first N tokens

        original_forward = self.talker.forward

        @functools.wraps(original_forward)
        def hooked_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)

            if hasattr(result, "hidden_states") and result.hidden_states is not None:
                hidden_states = result.hidden_states
                if isinstance(hidden_states, tuple) and len(hidden_states) == 2:
                    codec_ids = hidden_states[1]
                    if codec_ids is not None:
                        token_time = time.perf_counter()
                        if first_token_time[0] is None:
                            first_token_time[0] = token_time
                        # Track first 10 token times for analysis
                        if len(token_times) < 10:
                            token_times.append(token_time)
                        code_queue.put(codec_ids.detach().clone())

            return result

        def run_generation():
            try:
                self.talker.forward = hooked_forward

                self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt,
                )

            except Exception as e:
                generation_error[0] = e
            finally:
                self.talker.forward = original_forward
                generation_done.set()

        thread_start_time = time.perf_counter()
        threading.Thread(target=run_generation, daemon=True).start()

        codes = []
        prev_audio_len = 0
        frame_idx = 0
        frames_since_yield = 0
        first_decode_time = [None]

        while True:
            try:
                code = code_queue.get(timeout=0.005)
                codes.append(code)
                frame_idx += 1
                frames_since_yield += 1

                should_yield = (
                    frame_idx >= config.min_initial_frames and
                    frames_since_yield >= config.yield_every_n_frames
                )

                if not should_yield:
                    continue

                # Gated decode
                decode_start = time.perf_counter()
                with torch.cuda.stream(self.decode_stream):
                    all_codes = torch.stack(codes, dim=-1)
                    with torch.no_grad():
                        audio = self.decoder(all_codes)

                self.decode_stream.synchronize()
                
                if first_decode_time[0] is None:
                    first_decode_time[0] = time.perf_counter() - decode_start
                
                audio = audio.squeeze().float().cpu().numpy()

                if first_audio_time[0] is None:
                    first_audio_time[0] = time.perf_counter()

                new_audio = audio[prev_audio_len:]
                prev_audio_len = len(audio)

                elapsed = (time.perf_counter() - start_time) * 1000
                
                # Build detailed timing for first chunk
                meta = {
                    "frame_idx": frame_idx,
                    "elapsed_ms": elapsed,
                    "first_token_ms": (first_token_time[0] - start_time) * 1000 if first_token_time[0] else 0,
                    "first_audio_ms": (first_audio_time[0] - start_time) * 1000 if first_audio_time[0] else 0,
                }
                
                # Add detailed first-chunk breakdown
                if frame_idx == config.min_initial_frames:
                    meta["_first_chunk_breakdown"] = {
                        "prompt_load_ms": (prompt_load_time - start_time) * 1000,
                        "thread_start_ms": (thread_start_time - prompt_load_time) * 1000,
                        "prefill_ms": (first_token_time[0] - thread_start_time) * 1000 if first_token_time[0] else 0,
                        "tokens_1_to_4_ms": (token_times[3] - first_token_time[0]) * 1000 if len(token_times) >= 4 else 0,
                        "first_decode_ms": first_decode_time[0] * 1000 if first_decode_time[0] else 0,
                        "token_times_ms": [(t - start_time) * 1000 for t in token_times[:6]],
                    }

                yield new_audio, meta

                frames_since_yield = 0

            except queue.Empty:
                if generation_done.is_set():
                    break

        if generation_error[0]:
            raise generation_error[0]

        total_time = (time.perf_counter() - start_time) * 1000
        total_samples = prev_audio_len
        audio_duration = total_samples / self.sample_rate * 1000 if total_samples > 0 else 0

        yield np.array([], dtype=np.float32), {
            "is_final": True,
            "total_time_ms": total_time,
            "total_samples": total_samples,
            "audio_duration_ms": audio_duration,
            "rtf": total_time / audio_duration if audio_duration > 0 else 0,
        }


# -----------------------------
# Global singleton
# -----------------------------

_tts = None


def get_tts() -> SimpleStreamingTTS:
    global _tts
    if _tts is None:
        _tts = SimpleStreamingTTS()
    return _tts

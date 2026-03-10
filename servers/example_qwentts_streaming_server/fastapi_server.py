import os
import queue
import asyncio
import threading
import numpy as np
import json
import base64
import io
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import soundfile as sf
from typing import Generator

from qtts_sync_streaming import get_qsync_tts

model_lock = asyncio.Lock()
current_cancel: threading.Event | None = None

# Folder of text–audio pairs: audio as .wav/.mp3/.flac/etc., transcript as .txt (same stem = voice_id)
VOICES_DIR = os.environ.get("VOICES_DIR", os.path.join(os.path.dirname(__file__), "ref_samples"))
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

# Initialize Qsync TTS
tts = get_qsync_tts()

# voice_id (filename stem) -> registered voice path
VOICE_PATHS: dict[str, Path] = {}


def _load_voices_from_folder() -> None:
    """Scan VOICES_DIR for audio+txt pairs and register each; voice_id = filename stem."""
    voices_dir = Path(VOICES_DIR)
    if not voices_dir.is_dir():
        print(f"[TTS] VOICES_DIR not found: {VOICES_DIR} (set VOICES_DIR env or create ref_voices/)")
        return
    for p in sorted(voices_dir.iterdir()):
        if p.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        stem = p.stem
        txt_path = voices_dir / f"{stem}.txt"
        if not txt_path.is_file():
            print(f"[TTS] Skipping {p.name}: no matching {stem}.txt")
            continue
        try:
            transcript = txt_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[TTS] Skipping {p.name}: failed to read {txt_path}: {e}")
            continue
        if not transcript:
            print(f"[TTS] Skipping {p.name}: empty transcript")
            continue
        try:
            path = tts.register_voice(stem, str(p), transcript)
            VOICE_PATHS[stem] = path
            print(f"[TTS] Registered voice: {stem}")
        except Exception as e:
            print(f"[TTS] Failed to register {stem}: {e}")


_load_voices_from_folder()

# Fallback if no folder voices: optional single default (set REF_AUDIO/REF_TEXT env if needed)
DEFAULT_VOICE_PATH: Path | None = None
_ref_audio = os.environ.get("REF_AUDIO")
_ref_text = os.environ.get("REF_TEXT")
if _ref_audio and _ref_text and not VOICE_PATHS:
    print(f"[TTS] Registering default voice from REF_AUDIO...")
    DEFAULT_VOICE_PATH = tts.register_voice("default_voice", _ref_audio, _ref_text)
    VOICE_PATHS["default_voice"] = DEFAULT_VOICE_PATH

app = FastAPI(title="Qsync TTS API", version="1.0.0")


def audio_chunks_generator(
    input_text: str,
    voice_prompt_path: str,
    cancel_evt: threading.Event,
) -> Generator[str, None, None]:
    model_sr = tts.sample_rate

    # Stream inference: (audio_chunk, rig_params, meta); rig_params is always None
    for audio_chunk, _rig_params, meta in tts.generate_streaming(
        input_text,
        voice_prompt_path=voice_prompt_path,
        stop_event=cancel_evt,
    ):
        if cancel_evt.is_set():
            break
        if meta.get("is_final") or audio_chunk is None or len(audio_chunk) == 0:
            continue

        # Convert to int16 if necessary
        chunk = audio_chunk
        if chunk.dtype != np.int16:
            chunk = np.clip(chunk, -1.0, 1.0)
            chunk = (chunk * 32767.0).astype(np.int16)

        # Upsample 24k -> 48k to match the reference server behavior
        if model_sr == 24000:
            chunk_out = np.repeat(chunk, 2)
            sr_out = 48000
        else:
            chunk_out = chunk
            sr_out = model_sr

        # Create a WAV chunk (44 byte header + PCM)
        buf = io.BytesIO()
        sf.write(buf, chunk_out, sr_out, format='WAV', subtype='PCM_16')
        wav_data = buf.getvalue()

        # Encode to base64 and wrap in JSON for Inworld protocol
        b64_content = base64.b64encode(wav_data).decode('utf-8')
        result = {"result": {"audioContent": b64_content}}
        yield json.dumps(result) + "\n"


def _run_chunk_generator(
    input_text: str,
    voice_prompt_path: str,
    cancel_evt: threading.Event,
    thread_queue: queue.Queue,
) -> None:
    """Run the blocking generator in a thread; put chunks into thread_queue (blocks when full = backpressure)."""
    try:
        for chunk in audio_chunks_generator(
            input_text,
            voice_prompt_path,
            cancel_evt,
        ):
            if cancel_evt.is_set():
                break
            thread_queue.put(chunk)  # blocks if queue full, so producer slows down
    finally:
        thread_queue.put(None)  # sentinel


@app.get("/voices")
async def list_voices():
    """Return available voice IDs (filename stems from ref_voices folder)."""
    return {"voiceIds": list(VOICE_PATHS.keys())}


@app.post("/tts/v1/voice:stream")
async def create_speech(request: Request):
    """Compatible with Inworld TTS streaming protocol. Payload: text, voiceId (optional)."""
    global current_cancel

    data = await request.json()
    input_text = data.get("text", "")
    voice_id = (data.get("voiceId") or data.get("voice_id") or "").strip()

    if not input_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    # Resolve voice: voiceId (filename stem) -> registered path
    if voice_id and voice_id in VOICE_PATHS:
        voice_path = VOICE_PATHS[voice_id]
    elif not voice_id and DEFAULT_VOICE_PATH is not None:
        voice_path = DEFAULT_VOICE_PATH
    elif not voice_id and len(VOICE_PATHS) == 1:
        voice_path = next(iter(VOICE_PATHS.values()))
    else:
        available = list(VOICE_PATHS.keys()) or ["(none registered)"]
        raise HTTPException(
            status_code=400,
            detail=f"voiceId required or invalid. Available: {available}. Put audio+txt pairs in {VOICES_DIR}.",
        )

    # OVERRIDE: cancel the previous stream
    if current_cancel is not None:
        current_cancel.set()

    cancel_evt = threading.Event()
    current_cancel = cancel_evt

    async def stream():
        async with model_lock:
            loop = asyncio.get_running_loop()
            thread_queue = queue.Queue(maxsize=64)  # threading.Queue: put() blocks when full
            thread = threading.Thread(
                target=_run_chunk_generator,
                args=(input_text, str(voice_path), cancel_evt, thread_queue),
            )
            thread.start()
            try:
                while True:
                    # get() blocks; run in executor so we don't block the event loop
                    chunk = await loop.run_in_executor(None, thread_queue.get)
                    if chunk is None:
                        break
                    yield chunk
            finally:
                thread.join(timeout=5.0)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)

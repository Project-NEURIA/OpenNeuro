import os
import re
import time

import httpx
import ormsgpack


_CLIENT = httpx.Client(timeout=30.0)


def sanitize_filename(text, max_len=20):
    clean = re.sub(r"[^a-zA-Z0-9 ]", "", text)[:max_len].strip()
    return clean.replace(" ", "_") or "output"


def analyze_stream_timing(arrival_times, chunk_lengths, sr=44100):
    if not arrival_times:
        return None
        
    rows = []
    durs = [length / (2 * sr) for length in chunk_lengths]  # int16 = 2 bytes
    t0_play = arrival_times[0]
    
    buffer_audio = 0.0
    last_time = t0_play
    underruns = 0
    max_lead, min_lead = -1e9, 1e9
    
    for i, (t_arr, dur) in enumerate(zip(arrival_times, durs), start=1):
        if t_arr > t0_play:
            played = max(0.0, t_arr - last_time)
            buffer_audio -= played
            if buffer_audio < -1e-6:
                underruns += 1
                buffer_audio = 0.0
            last_time = t_arr
        
        buffer_audio += dur
        playback_pos = max(0.0, t_arr - t0_play)
        total_audio_received = sum(durs[:i])
        lead = total_audio_received - playback_pos
        
        max_lead = max(max_lead, lead)
        min_lead = min(min_lead, lead)
        
        rows.append({
            "chunk": i, "arrive_s": t_arr, "dur_s": dur,
            "lead_s": lead, "buffer_after_add_s": buffer_audio,
        })
    
    chunk2_before_chunk1_ends = arrival_times[1] <= (arrival_times[0] + durs[0]) if len(rows) >= 2 else None
    
    return {
        "t_play_start": t0_play, "underruns": underruns,
        "max_lead_s": max_lead, "min_lead_s": min_lead,
        "chunk2_before_chunk1_ends": chunk2_before_chunk1_ends,
        "rows": rows,
    }


def run_demo(
    text,
    *,
    url: str = "http://127.0.0.1:8082/v1/tts",
    reference_id: str | None = "female",
    use_memory_cache: str = "on",
    use_inline_reference: bool = False,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
):
    print(f"\n--- Processing: {text[:50]}... ---")
    
    references = []
    if use_inline_reference and ref_audio_path and os.path.exists(ref_audio_path):
        with open(ref_audio_path, "rb") as f:
            audio_data = f.read()
        references.append(
            {
                "audio": audio_data,
                "text": ref_text or "",
            }
        )
        # When using inline reference audio, let the server handle caching.
        reference_id = None

    data = {
        "text": text,
        "references": references,
        # For latency measurement, use raw PCM so every chunk is audio
        # and there is no separate WAV header.
        "format": "pcm",
        "streaming": True,
        "max_new_tokens": 1024,
        "chunk_length": 200,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
        "reference_id": reference_id,
        "use_memory_cache": use_memory_cache,
    }

    start_time = time.perf_counter()
    arrival_times = []
    audio_chunks = []
    chunk_lengths = []
    first_chunk_at = None

    # Reuse a persistent client across calls to avoid paying connection setup
    # on every benchmark request.
    with _CLIENT.stream(
        "POST",
        url,
        params={"format": "msgpack"},
        content=ormsgpack.packb(data),
        headers={
            "content-type": "application/msgpack",
        },
    ) as response:
        headers_received_at = time.perf_counter()

        if response.status_code != 200:
            print(f"Request failed: {response.status_code}")
            try:
                print(response.json())
            except Exception:
                pass
            return

        # With format="pcm", every non-empty chunk is actual audio bytes (int16 mono).
        for chunk in response.iter_bytes():
            if chunk:
                chunk_arrival = time.perf_counter()
                if first_chunk_at is None:
                    first_chunk_at = chunk_arrival
                arrival_times.append(chunk_arrival - start_time)
                audio_chunks.append(chunk)
                chunk_lengths.append(len(chunk))

    total_time = time.perf_counter() - start_time
    
    if not audio_chunks:
        print("No audio received")
        return

    output_dir = "demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sanitize_filename(text)}.wav"
    output_path = os.path.join(output_dir, filename)
    
    # The chunks are raw PCM int16 (from tools/server/inference.py)
    full_audio_bytes = b"".join(audio_chunks)
    
    # Duration from raw PCM: int16 mono @ 44100Hz
    bytes_per_sample = 2
    sample_rate = 44100
    num_samples = len(full_audio_bytes) // bytes_per_sample
    duration = num_samples / sample_rate if sample_rate > 0 else 0.0
    rtf = total_time / duration if duration > 0 else 0
    timing = analyze_stream_timing(arrival_times, chunk_lengths, sample_rate)

    print(f"Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Audio duration: {duration:.2f}s")
    print(f"  RTF: {rtf:.3f}")
    if timing:
        print(f"  Underruns: {timing['underruns']} | Min Lead: {timing['min_lead_s']*1000:.1f}ms")
        print(f"  Chunk 2 caught up: {timing['chunk2_before_chunk1_ends']}")

    print(f"  Response headers latency: {(headers_received_at - start_time)*1000:.1f}ms")
    if arrival_times:
        print(f"  First audio chunk latency: {arrival_times[0]*1000:.1f}ms")
        print(
            "  Headers -> first audio chunk: "
            f"{(first_chunk_at - headers_received_at)*1000:.1f}ms"
        )
    else:
        print("  First audio chunk latency: N/A")
    print(f"  Attempting to save {len(full_audio_bytes)} bytes to: {output_path}")
    
    import wave
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(full_audio_bytes)
        
    print(f"  Successfully saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    test_text = "Hello, this is a streaming test for Fish Speech S2 Pro [giggle]. I hope it sounds natural and has low latency."

    run_demo(test_text, reference_id="melina", use_memory_cache="on")

    long_text = "[whisper in small voice] I have a secret to tell you... [whisper in small voice] promise you won't tell anyone? [with strong british accent] I'm actually an AI [chuckle]! Ain't [emphasis]that fun?"
    run_demo(long_text, reference_id="melina", use_memory_cache="on")

    alt_text = "[悄悄话] 我有一个小秘密 想听吗? [哈哈] 我 其实是一只AI哦~ 是不是 [咳咳 清喉咙] 非常可爱?"
    run_demo(alt_text, reference_id="melina", use_memory_cache="on")

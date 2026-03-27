# Part 1 — Profiling Execution Time

## Overview

We profiled the **Time to First Audio (TTFA)** of our voice agent pipeline:

```
Microphone → VAD → ASR → LLM → TTS → Speaker
```

TTFA is the end-to-end latency from the user finishing speaking to the first byte of agent audio reaching the speaker. It is the primary quality metric for voice agents — humans expect responses within ~300ms; above ~800ms the conversation feels broken.

We profiled 4 key functions using Python's `cProfile` module. The profiler has two modes:
- **Sequential mode**: Runs each component one-at-a-time, feeding output → input. Each function gets its own dedicated `cProfile` session with no contention.
- **Pipeline mode**: Runs the full pipeline in parallel with wall-clock `time.perf_counter()` timestamps to measure real TTFA.


### Test Setup

- **Audio input**: Synthetic speech generated via macOS `say` (4.4s, 48kHz mono)
- **Pipeline**: FileSource → VAD → ASR → LLM → TTS → NullSink
- **ASR**: Groq Whisper API (`whisper-large-v3-turbo`)
- **LLM**: Groq `llama-3.3-70b-versatile` (default), also tested OpenAI `gpt-5.4`
- **TTS**: Inworld TTS API (`inworld-tts-1.5-max`)

---

## TTFA Results

| Metric | Value |
|--------|-------|
| **TTFA (VAD-end → first TTS audio)** | **1,330 ms** |

TTFA is measured from the moment VAD finalizes the user's speech segment (end-of-utterance) to the first TTS audio chunk arriving at the sink.

### TTFA Breakdown

Each stage is measured independently — no overlap or double-counting:

| Stage | Latency | What it measures |
|-------|---------|-----------------|
| ASR (speech-to-text) | 225 ms | VAD-end → ASR transcription complete |
| LLM TTFT (first token) | 768 ms | ASR-done → first LLM token streamed |
| TTS TTFB (first audio) | 336 ms | First LLM token → first TTS audio chunk |
| **TTFA** | **1,330 ms** | **VAD-end → first TTS audio (sum of above)** |

### Pipeline Hop Analysis

Using `pipeline_hop_test.py`, we confirmed there is no hidden overhead in the channel/thread infrastructure:

| Hop | Latency |
|-----|---------|
| VAD finalize (segment concat) | 1 ms |
| VAD → ASR channel transit | 0 ms |
| ASR → Adapter → LLM channel transit | 0 ms |


### LLM TTFT Comparison (Isolated)

Using `llm_ttft_test.py`, we measured raw LLM TTFT with no pipeline overhead:

| Model | Avg TTFT | Min | Max |
|-------|----------|-----|-----|
| Groq `llama-3.3-70b-versatile` | 390 ms | 257 ms | 646 ms |
| OpenAI `gpt-5.4` | 875 ms | 458 ms | 1,566 ms |
| OpenAI `gpt-4.1-nano` | 1,157 ms | 924 ms | 1,437 ms |

Groq is the fastest due to their custom LPU inference hardware. OpenAI models show higher variance.

---

## cProfile Analysis

### 1. `VAD._process_audio_frame` — 219 calls, 81ms total (0.4ms/call)

The VAD processes every 20ms audio chunk. Without Smart Turn (which only triggers during silence in the full pipeline), the hot path is Silero VAD inference:

| Function | Cumulative Time | Notes |
|----------|----------------|-------|
| `torch.nn.Module._call_impl` | 60 ms | Silero VAD neural network forward pass |
| `AudioFrame.get` (resampling) | 4 ms | numpy `interp` for 48kHz→16kHz |
| `torch.tensor` | 4 ms | Converting audio chunks for Silero |
| `deque.popleft` | 3 ms | VAD buffer management |

In the pipeline, Smart Turn ONNX inference adds significant overhead when it runs during silence detection (~13ms per chunk), but the core VAD loop itself is very fast.

### 2. `ASR._transcribe_audio` — 1 call, 553ms

Dominated entirely by the Groq Whisper HTTP API call:

| Function | Cumulative Time | Notes |
|----------|----------------|-------|
| `requests.post` → `sessions.send` | 550 ms | Full HTTP round-trip |
| `ssl._SSLSocket.read` | 465 ms | Waiting for Groq server response |
| `urllib3.connection.connect` | 72 ms | TLS handshake |
| `_prepare_audio_for_transcription` | ~3 ms | WAV file creation + resampling |

99% of time is network I/O.

### 3. `LLM` TTFT (`litellm.completion` → first token) — 540ms

cProfile captures the full `litellm.completion()` call up to the first streaming token:

| Function | Cumulative Time | Notes |
|----------|----------------|-------|
| `litellm.utils.wrapper` | 535 ms | litellm entry point |
| `token_counter` → `from_pretrained` | 312 ms | Tokenizer loading (one-time cost) |
| `litellm.main.completion` | 217 ms | Actual API call to Groq |
| `httpx.Client.send` | 198 ms | HTTP request + first response chunk |

**Key finding**: 312ms (58%) of the TTFT is litellm's tokenizer initialization (`from_pretrained`), not the actual API call. On subsequent calls this is cached and the real TTFT drops to ~200ms.

### 4. `TTS` `requests.post` (Inworld API) — 522ms

Profiles the Inworld TTS API call including streaming audio response:

| Function | Cumulative Time | Notes |
|----------|----------------|-------|
| `urllib3.connectionpool.urlopen` | 492 ms | Full HTTP round-trip + streaming |
| `ssl._SSLSocket.read` | 370 ms | Reading streaming audio chunks |
| `http.client._read_status` | 368 ms | Waiting for first response byte |
| `urllib3.connection.connect` | 122 ms | TLS handshake to Inworld |
| `create_connection` | 74 ms | TCP socket setup |

TLS handshake (122ms) is a significant fixed cost per request.

---

## Improvement Suggestions

### 1. `LLM` TTFT — Warm up tokenizer and use prompt caching (768ms, 58% of TTFA)

**Problem**: LLM TTFT is the single largest contributor to TTFA. cProfile reveals 312ms of the 540ms is litellm loading the tokenizer via `from_pretrained`. The actual API call is only ~200ms.

**Improvement options**:
- **Warm up tokenizer**: Call `litellm.completion()` once at startup (with a dummy prompt) so the tokenizer is cached before real requests. This alone saves ~312ms on the first call.
- **Prompt caching**: Groq and OpenAI support prompt caching for repeated system prompts. The system message is identical every turn — caching avoids re-processing it.
- **Smaller model**: `llama-3.1-8b` on Groq has ~100ms TTFT vs ~390ms for 70b.

### 2. `TTS` `requests.post` — Pre-warm TLS connection (336ms, 25% of TTFA)

**Problem**: TLS handshake to Inworld costs 122ms per request. Each TTS sentence creates a new connection.

**Improvement options**:
- **Persistent connection**: Use `requests.Session()` with connection pooling to reuse the TLS session across requests, eliminating the 122ms handshake after the first call.
- **Smaller chunk threshold**: Send to TTS after a clause boundary (comma, dash) or after N tokens, not just sentence boundaries. This trades naturalness for speed.
- **Filler audio**: Play a brief filler sound ("Let me think...") immediately while waiting for the first real TTS chunk.

### 3. `ASR._transcribe_audio` — Use streaming ASR or local model (225ms, 17% of TTFA)

**Problem**: Each API call to Groq Whisper takes ~550ms, and it processes the entire segment at once (batch, not streaming).

**Improvement options**:
- **Streaming ASR**: Use a WebSocket-based ASR service (e.g., Deepgram, AssemblyAI) that transcribes incrementally as audio arrives, eliminating the segment-level batch delay.
- **Local Whisper**: Run `whisper.cpp` or `faster-whisper` locally. For short utterances (<3s), local inference on Apple Silicon can be faster than a network round-trip.

### 4. `VAD._process_audio_frame` — Reduce Smart Turn frequency (<5ms/chunk, minimal TTFA impact)

**Problem**: In the pipeline, `_check_smart_turn()` runs ONNX inference (~13ms each) on every audio chunk during silence detection. This doesn't directly affect TTFA (it runs before end-of-utterance), but it wastes CPU.

**Improvement**: Only run Smart Turn every N chunks (e.g., every 200ms instead of every 20ms). The turn detection doesn't need 50Hz resolution — checking 5x/second is sufficient. This would reduce Smart Turn overhead by ~10x.

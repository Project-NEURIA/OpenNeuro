"""Quick standalone LLM TTFT test — no pipeline, just litellm.completion()."""

import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from litellm import completion  # noqa: E402

MODELS = [
    "groq/llama-3.3-70b-versatile",
    "openai/gpt-5.4",
    "openai/gpt-4.1-nano",
]

MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful voice assistant. Keep responses brief.",
    },
    {"role": "user", "content": "Hello, how are you doing today?"},
]


def test_ttft(model: str, runs: int = 3):
    print(f"\n{'=' * 60}")
    print(f"Model: {model}  ({runs} runs)")
    print(f"{'=' * 60}")

    ttfts = []
    for i in range(runs):
        t0 = time.perf_counter()
        stream = completion(model=model, messages=MESSAGES, stream=True)

        first_token_text = None
        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice and choice.delta and choice.delta.content:
                t_first = time.perf_counter()
                first_token_text = choice.delta.content
                break

        if first_token_text is not None:
            ttft_ms = (t_first - t0) * 1000
            ttfts.append(ttft_ms)
            # Drain the rest
            for _ in stream:
                pass
            print(
                f"  Run {i + 1}: TTFT = {ttft_ms:.0f}ms  (first token: {first_token_text!r})"
            )
        else:
            print(f"  Run {i + 1}: No tokens received")

    if ttfts:
        print(
            f"  Avg TTFT: {sum(ttfts) / len(ttfts):.0f}ms  Min: {min(ttfts):.0f}ms  Max: {max(ttfts):.0f}ms"
        )


if __name__ == "__main__":
    for model in MODELS:
        try:
            test_ttft(model)
        except Exception as e:
            print(f"  ERROR: {e}")

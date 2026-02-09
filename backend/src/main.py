from __future__ import annotations

from dotenv import load_dotenv

DEFAULT_CONFIG = {
    "nodes": ["Mic", "STS", "Speaker"],
    "edges": [
        {"source": "Mic", "target": "STS"},
        {"source": "STS", "target": "Speaker"},
    ],
}


def main() -> None:
    load_dotenv(override=True)

    from .server import app, manager

    # Apply default pipeline (Mic → STS → Speaker)
    result = manager.apply(DEFAULT_CONFIG)
    print(f"[backend] Default pipeline: {result}")

    import uvicorn
    print("[backend] API server starting on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()

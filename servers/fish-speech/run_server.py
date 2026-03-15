import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

# Global cache directory
CACHE_DIR = Path("/virtual/my_tmp/fish/checkpoints")
MODEL_NAME = "fishaudio/s2-pro"
CHECKPOINT_PATH = CACHE_DIR / "s2-pro"

def download_model():
    print("here")
    if not CHECKPOINT_PATH.exists():
        print(f"Downloading model {MODEL_NAME} to {CHECKPOINT_PATH}...")
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=CHECKPOINT_PATH,
            local_dir_use_symlinks=False
        )
    else:
        print(f"Model already exists at {CHECKPOINT_PATH}")

def run_server():
    download_model()
    
    # Paths for the server
    llama_checkpoint = CHECKPOINT_PATH
    decoder_checkpoint = CHECKPOINT_PATH / "codec.pth"
    
    cmd = [
        "python", "tools/api_server.py",
        "--llama-checkpoint-path", str(llama_checkpoint),
        "--decoder-checkpoint-path", str(decoder_checkpoint),
        "--listen", "0.0.0.0:8082",
        "--half",
        "--workers", "1",
        "--load-4bit",
        "--compile",
        "--streaming-chunk-size", "4",
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    # Use relative path to get the directory where this script is located
    fish_speech_dir = Path(__file__).parent.resolve()
    
    # Add the current directory to PYTHONPATH so we don't need to pip install -e .
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fish_speech_dir) + os.pathsep + env.get("PYTHONPATH", "")
    
    subprocess.run(cmd, cwd=fish_speech_dir, env=env)

if __name__ == "__main__":
    print("Starting FishTTS API server...")
    run_server()

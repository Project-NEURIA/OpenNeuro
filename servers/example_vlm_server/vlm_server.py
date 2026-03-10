import os
import requests
import json
import io
import base64
import random
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VLLM_URL = "http://localhost:8055/v1/chat/completions"
MODEL_ID = "Qwen/Qwen3.5-0.8B"

VLM_FPS = 1
WINDOW_DURATION = 3.0
TOTAL_DURATION = 6.0 # 2 windows of 3 seconds
MEMORY_SIZE = 4      # Number of previous window captions to keep in history

# Prompts from synth_sft_gen.py
PROMPT_WINDOW1 = (
    "You are an advanced real-time vision module for blind people. "
    "Given the current observation, use short phrases to caption what you see, include movement and composition if needed. "
    "Keep it short, efficient, real-time, relavant. Think \"What would the blink person like to know?\""
)

PROMPT_WINDOW2_TEMPLATE = (
    "You are an advanced real-time vision module for blind people. "
    "Given the current observation history, generate an efficient delta caption "
    "only introducing new observations that were not mentioned in the history context. "
    "Keep it efficient, no redundant information, Think \"What would the blink person like to know?\"\nDO NOT REPEAT anything that is already mentioned.\n\n"
    "History context:\n{context}"
)

def encode_pil_to_base64_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def download_video(url: str, save_path: str):
    print(f"Downloading video from {url}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {save_path}")

def sample_video_frames_entire(video_path: str, fps: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames_in_video / video_fps
    
    total_samples = int(video_duration * fps)
    
    frames = []
    for i in range(total_samples):
        target_time = i / fps
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        else:
            break
            
    cap.release()
    return frames

def call_vllm(frames: List[Image.Image], prompt: str) -> str:
    # Build content with multiple images and text
    content = []
    for img in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_pil_to_base64_uri(img)}
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
        "temperature": 0.2,
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(VLLM_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()
        
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

def run_pipeline(video_url: str):
    video_path = "temp_video.mp4"
    try:
        # 1. Download
        download_video(video_url, video_path)
        
        # 2. Sample entire video
        all_frames = sample_video_frames_entire(video_path, VLM_FPS)
        
        frames_per_window = int(WINDOW_DURATION * VLM_FPS)
        num_windows = len(all_frames) // frames_per_window
        
        captions_history = []
        
        for i in range(num_windows):
            start_idx = i * frames_per_window
            end_idx = start_idx + frames_per_window
            window_frames = all_frames[start_idx:end_idx]
            
            print(f"\n--- Processing Window {i+1} ({i*WINDOW_DURATION}s - {(i+1)*WINDOW_DURATION}s) ---")
            
            if i == 0:
                # First window uses the initial prompt
                caption = call_vllm(window_frames, PROMPT_WINDOW1)
            else:
                # Subsequent windows use the delta prompt with history
                history_str = " ".join(captions_history)
                prompt = PROMPT_WINDOW2_TEMPLATE.format(context=history_str)
                caption = call_vllm(window_frames, prompt)
            
            print(f"Window {i+1} Result: {caption}")
            
            # Update history with sliding window
            captions_history.append(caption)
            if len(captions_history) > MEMORY_SIZE:
                captions_history.pop(0)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    test_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/video/N1cdUjctpG8.mp4"
    run_pipeline(test_url)

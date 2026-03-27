"""Run pipeline mode N times and print averaged TTFA breakdown."""

import subprocess
import re
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

asr_times = []
llm_times = []
tts_times = []
ttfa_times = []

for i in range(N):
    print(f"\n{'='*40} Run {i+1}/{N} {'='*40}")
    result = subprocess.run(
        [sys.executable, "-m", "profiling.ttfa_profile", "--mode", "pipeline"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent),
        timeout=120,
    )
    output = result.stdout + result.stderr

    asr = re.search(r"ASR latency.*?:\s+(\d+)ms", output)
    llm = re.search(r"LLM TTFT.*?:\s+(\d+)ms", output)
    tts = re.search(r"TTS TTFB.*?:\s+(\d+)ms", output)
    ttfa = re.search(r"TTFA.*?:\s+(\d+)ms", output)

    if asr and llm and tts and ttfa:
        asr_times.append(int(asr.group(1)))
        llm_times.append(int(llm.group(1)))
        tts_times.append(int(tts.group(1)))
        ttfa_times.append(int(ttfa.group(1)))
        print(f"  ASR={asr.group(1)}ms  LLM={llm.group(1)}ms  TTS={tts.group(1)}ms  TTFA={ttfa.group(1)}ms")
    else:
        print(f"  FAILED to parse output")

print(f"\n{'='*60}")
print(f"RESULTS ({len(ttfa_times)}/{N} successful runs)")
print(f"{'='*60}")
if ttfa_times:
    print(f"  ASR:  avg={sum(asr_times)//len(asr_times)}ms  min={min(asr_times)}ms  max={max(asr_times)}ms")
    print(f"  LLM:  avg={sum(llm_times)//len(llm_times)}ms  min={min(llm_times)}ms  max={max(llm_times)}ms")
    print(f"  TTS:  avg={sum(tts_times)//len(tts_times)}ms  min={min(tts_times)}ms  max={max(tts_times)}ms")
    print(f"  TTFA: avg={sum(ttfa_times)//len(ttfa_times)}ms  min={min(ttfa_times)}ms  max={max(ttfa_times)}ms")

#!/bin/bash
# Run pipeline mode N times in separate processes and average results.
# Usage: bash profiling/bench_pipeline.sh [N]

N=${1:-10}
DIR="$(cd "$(dirname "$0")/.." && pwd)"

asr_sum=0; llm_sum=0; tts_sum=0; ttfa_sum=0; ok=0

for i in $(seq 1 $N); do
    echo "======== Run $i/$N ========"
    output=$(cd "$DIR" && uv run --no-group cuda python -m profiling.ttfa_profile --mode pipeline 2>&1)

    asr=$(echo "$output" | sed -n 's/.*ASR latency.*:[[:space:]]*\([0-9]*\)ms.*/\1/p')
    llm=$(echo "$output" | sed -n 's/.*LLM TTFT.*:[[:space:]]*\([0-9]*\)ms.*/\1/p')
    tts=$(echo "$output" | sed -n 's/.*TTS TTFB.*:[[:space:]]*\([0-9]*\)ms.*/\1/p')
    ttfa=$(echo "$output" | sed -n 's/.*TTFA.*:[[:space:]]*\([0-9]*\)ms.*/\1/p')

    if [ -n "$asr" ] && [ -n "$llm" ] && [ -n "$tts" ] && [ -n "$ttfa" ]; then
        echo "  ASR=${asr}ms  LLM=${llm}ms  TTS=${tts}ms  TTFA=${ttfa}ms"
        asr_sum=$((asr_sum + asr))
        llm_sum=$((llm_sum + llm))
        tts_sum=$((tts_sum + tts))
        ttfa_sum=$((ttfa_sum + ttfa))
        ok=$((ok + 1))
    else
        echo "  FAILED"
    fi
done

echo ""
echo "============ RESULTS ($ok/$N) ============"
if [ $ok -gt 0 ]; then
    echo "  ASR avg:  $((asr_sum / ok))ms"
    echo "  LLM avg:  $((llm_sum / ok))ms"
    echo "  TTS avg:  $((tts_sum / ok))ms"
    echo "  TTFA avg: $((ttfa_sum / ok))ms"
fi

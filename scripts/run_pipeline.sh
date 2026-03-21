#!/usr/bin/env bash
# run_pipeline.sh — End-to-end VLA data pipeline
#
# Usage:
#   bash scripts/run_pipeline.sh <session_dir> [options]
#
# Options:
#   --scenario   delivery|driving|warehouse|kitchen  (default: unknown)
#   --skip       comma-separated stages to skip
#   --upload-hf  push episodes to HuggingFace after chunking
#   --hf-repo    HuggingFace dataset repo id
#   --no-resume  start fresh even if checkpoint exists
#   --label      push frames to LabelStudio after pipeline
#
# Example:
#   bash scripts/run_pipeline.sh /data/sessions/delivery-2026-03-18 \
#       --scenario delivery --upload-hf --hf-repo myuser/india-vla-v1

set -euo pipefail

SESSION_DIR=""
SCENARIO="unknown"
SKIP_STAGES=""
UPLOAD_HF=false
HF_REPO=""
RESUME="--resume"
PUSH_LABELS=false

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <session_dir> [--scenario TYPE] [--skip STAGES] [--upload-hf] [--hf-repo REPO] [--no-resume] [--label]"
    exit 1
fi

SESSION_DIR="$1"; shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario)   SCENARIO="$2";       shift 2 ;;
        --skip)       SKIP_STAGES="$2";    shift 2 ;;
        --upload-hf)  UPLOAD_HF=true;      shift   ;;
        --hf-repo)    HF_REPO="$2";        shift 2 ;;
        --no-resume)  RESUME="--no-resume"; shift  ;;
        --label)      PUSH_LABELS=true;    shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$SESSION_DIR" ]]; then
    echo "ERROR: Session directory not found: $SESSION_DIR"; exit 1
fi

SESSION_ID=$(basename "$SESSION_DIR")
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  India VLA Pipeline"
echo "  Session:  $SESSION_ID"
echo "  Scenario: $SCENARIO"
echo "  Dir:      $SESSION_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

[[ -f ".venv/bin/activate" ]] && source .venv/bin/activate || true
[[ -f "venv/bin/activate"  ]] && source venv/bin/activate  || true

CMD="python -m src.pipeline.runner run \"$SESSION_DIR\" $RESUME --scenario $SCENARIO"
[[ -n "$SKIP_STAGES" ]] && CMD="$CMD --skip $SKIP_STAGES"
[[ "$UPLOAD_HF" == true ]] && CMD="$CMD --upload-hf"
[[ -n "$HF_REPO" ]] && CMD="$CMD --hf-repo $HF_REPO"

echo "Running: $CMD"
eval "$CMD"
PIPELINE_EXIT=$?

if [[ $PIPELINE_EXIT -ne 0 ]]; then
    echo "❌ Pipeline failed (exit $PIPELINE_EXIT)"; exit $PIPELINE_EXIT
fi

if [[ "$PUSH_LABELS" == true ]]; then
    echo "Pushing frames to LabelStudio..."
    API_URL="${API_URL:-http://localhost:8000}"
    curl -s -X POST "$API_URL/labeling/push/$SESSION_ID" \
        -H "Content-Type: application/json" \
        -d "{\"project_title\":\"india-vla-$SCENARIO\",\"auto_label\":true,\"use_segments\":true}" \
        | python3 -m json.tool || true
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Pipeline complete: $SESSION_ID"
EPISODES_DIR="$SESSION_DIR/episodes"
[[ -d "$EPISODES_DIR" ]] && echo "  Episodes: $(find "$EPISODES_DIR" -name '*.h5' | wc -l | tr -d ' ')"
REPORT="$SESSION_DIR/assembled/validation_report.json"
if [[ -f "$REPORT" ]]; then
    python3 -c "import json; r=json.load(open('$REPORT')); print(f'  Valid: {r[\"valid_count\"]}/{r[\"total_count\"]}')" 2>/dev/null || true
fi
echo "  Output: $SESSION_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

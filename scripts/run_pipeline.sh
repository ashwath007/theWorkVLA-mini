#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh — End-to-end VLA data processing pipeline
# ============================================================================
# Usage:
#   bash scripts/run_pipeline.sh <session_directory> [--upload]
#
# Example:
#   bash scripts/run_pipeline.sh /data/sessions/2024-01-15-1030/abc123
#   bash scripts/run_pipeline.sh /data/sessions/2024-01-15-1030/abc123 --upload
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
success() { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date +%H:%M:%S)] !${NC} $*"; }
error()   { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*" >&2; exit 1; }

step_header() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  STEP $1: $2${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ── Arguments ─────────────────────────────────────────────────────────────────
SESSION_DIR="${1:-}"
UPLOAD_FLAG="${2:-}"

if [[ -z "$SESSION_DIR" ]]; then
    echo "Usage: $0 <session_directory> [--upload]"
    echo ""
    echo "Example:"
    echo "  $0 /data/sessions/2024-01-15-1030/abc123"
    echo "  $0 /data/sessions/2024-01-15-1030/abc123 --upload"
    exit 1
fi

SESSION_DIR="$(realpath "$SESSION_DIR")"
[[ -d "$SESSION_DIR" ]] || error "Session directory not found: $SESSION_DIR"

METADATA="$SESSION_DIR/metadata.json"
[[ -f "$METADATA" ]] || error "metadata.json not found in $SESSION_DIR"

# Load session ID
SESSION_ID=$(python3 -c "import json; print(json.load(open('$METADATA'))['session_id'])" 2>/dev/null || basename "$SESSION_DIR")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
EPISODES_DIR="$SESSION_DIR/episodes"

# Activate virtualenv if present
if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    info "Activated virtual environment."
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║   India VLA Data Pipeline                  ║${NC}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
info "Session ID  : $SESSION_ID"
info "Session dir : $SESSION_DIR"
info "Upload      : ${UPLOAD_FLAG:---}"

PIPELINE_START=$(date +%s)

# ── Step 1: Validate inputs ───────────────────────────────────────────────────
step_header "1" "Validate inputs"

required_files=("video.mp4" "audio.wav" "imu.csv" "metadata.json")
missing=0
for f in "${required_files[@]}"; do
    if [[ ! -f "$SESSION_DIR/$f" ]]; then
        warn "Missing: $f"
        missing=$((missing + 1))
    else
        success "Found: $f"
    fi
done

if [[ $missing -gt 0 ]]; then
    warn "$missing required file(s) missing — some steps may be skipped."
fi

# ── Step 2: Preprocess ────────────────────────────────────────────────────────
step_header "2" "Preprocess (sync + face blur + HDF5)"

HDF5_PATH="$SESSION_DIR/session.hdf5"
if [[ -f "$HDF5_PATH" ]]; then
    warn "session.hdf5 already exists — skipping preprocessing."
else
    info "Running stream synchronization and preprocessing …"
    $PYTHON - << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])

import json, numpy as np
from pathlib import Path

session_dir = Path(os.environ["SESSION_DIR"])
meta = json.loads((session_dir / "metadata.json").read_text())

from preprocess.sync import StreamSynchronizer
from preprocess.video import VideoPreprocessor
from preprocess.audio import AudioPreprocessor
from preprocess.imu import IMUPreprocessor
from preprocess.hdf5_writer import HDF5Writer

print("  Aligning streams …")
syncer = StreamSynchronizer()
ts_csv = session_dir / "video_timestamps.csv"
aligned = syncer.align_streams(
    str(session_dir / "video.mp4"),
    str(session_dir / "audio.wav"),
    str(session_dir / "imu.csv"),
    str(session_dir / "metadata.json"),
    video_timestamps_csv=str(ts_csv) if ts_csv.exists() else None,
)

print("  Preprocessing video frames (face blur) …")
vp = VideoPreprocessor()
frames, _ = vp.preprocess_pipeline(str(session_dir / "video.mp4"), apply_face_blur=True)

print("  Preprocessing audio …")
ap = AudioPreprocessor()
audio_chunks, _ = ap.preprocess(str(session_dir / "audio.wav"), aligned["frame_timestamps"])
audio_arr = np.array(audio_chunks, dtype="float32")

print("  Preprocessing IMU …")
imu_proc = IMUPreprocessor()
imu_df, quats = imu_proc.preprocess_pipeline(str(session_dir / "imu.csv"), target_fps=30)
imu_arr = imu_df[["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]].to_numpy(dtype="float32")
if quats is not None:
    imu_arr = np.concatenate([imu_arr, quats], axis=1)

T = min(len(frames), len(audio_arr), len(imu_arr))
print(f"  Writing HDF5: {T} frames …")
writer = HDF5Writer()
writer.write_session(
    session_id=meta["session_id"],
    frames=frames[:T],
    audio_chunks=audio_arr[:T],
    imu_data=imu_arr[:T],
    metadata=meta,
    output_path=str(session_dir / "session.hdf5"),
)
print("  Done.")
PYEOF
    success "Preprocessing complete → $HDF5_PATH"
fi

SESSION_DIR="$SESSION_DIR" # re-export for subshell
export SESSION_DIR

# ── Step 3: Segmentation ──────────────────────────────────────────────────────
step_header "3" "Action segmentation (optical flow)"

SEGMENTS_PATH="$SESSION_DIR/segments.json"
if [[ -f "$SEGMENTS_PATH" ]]; then
    warn "segments.json already exists — skipping segmentation."
else
    info "Running optical flow segmentation …"
    $PYTHON - << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])

import json, h5py, cv2, numpy as np
from pathlib import Path

session_dir = Path(os.environ["SESSION_DIR"])
hdf5_path   = session_dir / "session.hdf5"

with h5py.File(str(hdf5_path), "r") as f:
    frames_arr = f["observation/images/front_camera"][:]

bgr_frames = [
    cv2.cvtColor((f * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    for f in frames_arr
]

from segmentation.action_segmenter import ActionSegmenter
segmenter = ActionSegmenter()
print(f"  Computing flow for {len(bgr_frames)} frames …")
segments = segmenter.segment_optical_flow(bgr_frames)
print(f"  Found {len(segments)} segments.")

seg_data = [s.to_dict() for s in segments]
(session_dir / "segments.json").write_text(json.dumps(seg_data, indent=2))
print("  Done.")
PYEOF
    SEG_COUNT=$(python3 -c "import json; d=json.load(open('$SEGMENTS_PATH')); print(len(d))")
    success "Segmentation complete: $SEG_COUNT segments → $SEGMENTS_PATH"
fi

# ── Step 4: LeRobot chunking ──────────────────────────────────────────────────
step_header "4" "LeRobot chunking"

if [[ -d "$EPISODES_DIR" && "$(ls -A "$EPISODES_DIR" 2>/dev/null)" ]]; then
    warn "Episodes directory already exists — skipping chunking."
else
    info "Generating LeRobot episodes …"
    $PYTHON - << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])

import json
from pathlib import Path

session_dir  = Path(os.environ["SESSION_DIR"])
hdf5_path    = session_dir / "session.hdf5"
segments_path = session_dir / "segments.json"
episodes_dir = session_dir / "episodes"

seg_dicts = json.loads(segments_path.read_text())

from segmentation.action_segmenter import ActionSegment
from segmentation.lerobot_chunker import LeRobotChunker

segments = [
    ActionSegment(
        start_frame=s["start_frame"], end_frame=s["end_frame"],
        confidence=s["confidence"], motion_magnitude=s["motion_magnitude"],
    )
    for s in seg_dicts
]

chunker  = LeRobotChunker()
episodes = list(chunker.chunk_session(str(hdf5_path), segments, []))
saved    = chunker.save_episodes(episodes, str(episodes_dir))
chunker.create_dataset_info(episodes, str(episodes_dir))
print(f"  Saved {len(saved)} episodes to {episodes_dir}")
PYEOF
    EP_COUNT=$(ls "$EPISODES_DIR"/*.hdf5 2>/dev/null | wc -l || echo 0)
    success "Chunking complete: $EP_COUNT episodes → $EPISODES_DIR"
fi

# ── Step 5: Optional upload ───────────────────────────────────────────────────
if [[ "$UPLOAD_FLAG" == "--upload" ]]; then
    step_header "5" "Upload to Hugging Face Hub"

    HF_TOKEN="${HF_TOKEN:-}"
    HF_DATASET_REPO="${HF_DATASET_REPO:-}"

    if [[ -z "$HF_TOKEN" || -z "$HF_DATASET_REPO" ]]; then
        error "HF_TOKEN and HF_DATASET_REPO must be set to upload."
    fi

    info "Uploading episodes to $HF_DATASET_REPO …"
    $PYTHON - << 'PYEOF'
import sys, os
sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])

from storage.hf_uploader import HFUploader

uploader = HFUploader(
    repo_id=os.environ["HF_DATASET_REPO"],
    token=os.environ["HF_TOKEN"],
)
urls = uploader.upload_dataset(os.environ["SESSION_DIR"] + "/episodes")
print(f"  Uploaded {len(urls)} files.")
PYEOF
    success "Upload complete."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
ELAPSED=$((PIPELINE_END - PIPELINE_START))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║   Pipeline Complete                        ║${NC}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
info "Session ID : $SESSION_ID"
info "Duration   : ${MINUTES}m ${SECONDS}s"
info "Output dir : $SESSION_DIR"
echo ""

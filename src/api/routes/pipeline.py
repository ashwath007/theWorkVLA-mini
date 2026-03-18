"""
Pipeline router: trigger preprocessing, segmentation, and chunking jobs.

All heavy processing runs as FastAPI BackgroundTasks.  Job status is tracked
in an in-memory dict (suitable for single-process deployments; swap for Redis
or a DB-backed store for multi-worker setups).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Job store (in-memory) ─────────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}


def _new_job(session_id: str, job_type: str) -> str:
    """Create a job entry and return its ID."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":     job_id,
        "session_id": session_id,
        "type":       job_type,
        "status":     "queued",
        "message":    "",
        "progress":   0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs, updated_at=time.time())


# ── Schemas ───────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    session_id: str
    type: str
    status: str          # queued | running | completed | failed
    message: str
    progress: int        # 0–100
    created_at: float
    updated_at: float


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_data_dir(request: Request) -> str:
    return getattr(request.app.state, "data_dir", os.environ.get("DATA_DIR", "/data/sessions"))


def _find_session_dir(data_dir: str, session_id: str) -> Optional[Path]:
    """Locate the session directory by scanning for metadata.json."""
    for p in Path(data_dir).rglob("metadata.json"):
        try:
            meta = json.loads(p.read_text())
            if meta.get("session_id") == session_id:
                return p.parent
        except Exception:
            pass
    return None


# ── Background workers ────────────────────────────────────────────────────────

def _run_preprocess(job_id: str, session_dir: Path) -> None:
    """Full preprocessing pipeline: sync streams → HDF5."""
    try:
        _update_job(job_id, status="running", message="Starting preprocessing …", progress=5)

        video_path  = session_dir / "video.mp4"
        audio_path  = session_dir / "audio.wav"
        imu_path    = session_dir / "imu.csv"
        meta_path   = session_dir / "metadata.json"
        ts_csv      = session_dir / "video_timestamps.csv"

        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json missing in {session_dir}")

        from src.preprocess.sync import StreamSynchronizer
        from src.preprocess.video import VideoPreprocessor
        from src.preprocess.audio import AudioPreprocessor
        from src.preprocess.imu import IMUPreprocessor
        from src.preprocess.hdf5_writer import HDF5Writer

        syncer  = StreamSynchronizer()
        aligned = syncer.align_streams(
            str(video_path), str(audio_path), str(imu_path), str(meta_path),
            video_timestamps_csv=str(ts_csv) if ts_csv.exists() else None,
        )
        _update_job(job_id, message="Streams aligned", progress=30)

        vp   = VideoPreprocessor()
        frames_array, _ = vp.preprocess_pipeline(
            str(video_path), apply_face_blur=True
        )
        _update_job(job_id, message="Frames preprocessed", progress=55)

        ap = AudioPreprocessor()
        audio_chunks, _ = ap.preprocess(
            str(audio_path), aligned["frame_timestamps"]
        )
        import numpy as np
        audio_arr = np.array(audio_chunks, dtype="float32")

        _update_job(job_id, message="Audio preprocessed", progress=70)

        imu_proc = IMUPreprocessor()
        imu_df, quats = imu_proc.preprocess_pipeline(str(imu_path), target_fps=30)
        imu_arr = imu_df[["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype="float32")
        if quats is not None:
            imu_arr = np.concatenate([imu_arr, quats], axis=1)
        T = min(len(frames_array), len(audio_arr), len(imu_arr))
        imu_arr = imu_arr[:T]

        _update_job(job_id, message="IMU preprocessed", progress=80)

        meta = json.loads(meta_path.read_text())
        hdf5_out = session_dir / "session.hdf5"
        writer   = HDF5Writer()
        writer.write_session(
            session_id=meta.get("session_id", "?"),
            frames=frames_array[:T],
            audio_chunks=audio_arr[:T],
            imu_data=imu_arr[:T],
            metadata=meta,
            output_path=str(hdf5_out),
        )
        _update_job(job_id, status="completed", message=f"HDF5 saved: {hdf5_out}", progress=100)
        logger.info("Preprocessing complete for %s", session_dir)

    except Exception as exc:
        _update_job(job_id, status="failed", message=str(exc), progress=0)
        logger.exception("Preprocessing failed for job %s", job_id)


def _run_segment(job_id: str, session_dir: Path) -> None:
    """Optical flow segmentation of preprocessed frames."""
    try:
        _update_job(job_id, status="running", message="Segmenting actions …", progress=10)

        import numpy as np
        import h5py, json

        hdf5_path = session_dir / "session.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError("session.hdf5 not found. Run preprocess first.")

        with h5py.File(str(hdf5_path), "r") as f:
            frames = f["observation/images/front_camera"][:]

        # Convert float32 [0,1] → uint8 BGR for optical flow
        import cv2
        bgr_frames = [
            cv2.cvtColor((f * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
            for f in frames
        ]

        from src.segmentation.action_segmenter import ActionSegmenter
        segmenter = ActionSegmenter()
        _update_job(job_id, message="Computing optical flow …", progress=40)
        segments = segmenter.segment_optical_flow(bgr_frames)

        seg_data = [s.to_dict() for s in segments]
        out_path = session_dir / "segments.json"
        out_path.write_text(json.dumps(seg_data, indent=2))

        _update_job(
            job_id,
            status="completed",
            message=f"{len(segments)} segments saved to {out_path}",
            progress=100,
        )
    except Exception as exc:
        _update_job(job_id, status="failed", message=str(exc), progress=0)
        logger.exception("Segmentation failed for job %s", job_id)


def _run_chunk(job_id: str, session_dir: Path) -> None:
    """Chunk segments into LeRobot episodes and save HDF5 per episode."""
    try:
        _update_job(job_id, status="running", message="Chunking episodes …", progress=10)
        import json

        hdf5_path    = session_dir / "session.hdf5"
        segments_path = session_dir / "segments.json"

        if not hdf5_path.exists():
            raise FileNotFoundError("session.hdf5 not found. Run preprocess first.")
        if not segments_path.exists():
            raise FileNotFoundError("segments.json not found. Run segment first.")

        seg_dicts = json.loads(segments_path.read_text())

        from src.segmentation.action_segmenter import ActionSegment
        from src.segmentation.lerobot_chunker import LeRobotChunker

        segments = [
            ActionSegment(
                start_frame=s["start_frame"],
                end_frame=s["end_frame"],
                confidence=s["confidence"],
                motion_magnitude=s["motion_magnitude"],
            )
            for s in seg_dicts
        ]

        chunker      = LeRobotChunker()
        episodes_dir = session_dir / "episodes"
        episodes     = list(chunker.chunk_session(str(hdf5_path), segments, []))
        saved        = chunker.save_episodes(episodes, str(episodes_dir))
        chunker.create_dataset_info(episodes, str(episodes_dir))

        _update_job(
            job_id,
            status="completed",
            message=f"{len(saved)} episodes saved to {episodes_dir}",
            progress=100,
        )
    except Exception as exc:
        _update_job(job_id, status="failed", message=str(exc), progress=0)
        logger.exception("Chunking failed for job %s", job_id)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/preprocess/{session_id}", status_code=202)
async def preprocess_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    data_dir: str = Depends(get_data_dir),
) -> dict:
    """Trigger preprocessing pipeline for a session (async)."""
    sdir = _find_session_dir(data_dir, session_id)
    if sdir is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    job_id = _new_job(session_id, "preprocess")
    background_tasks.add_task(_run_preprocess, job_id, sdir)
    return {"job_id": job_id, "status": "queued"}


@router.post("/segment/{session_id}", status_code=202)
async def segment_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    data_dir: str = Depends(get_data_dir),
) -> dict:
    """Trigger action segmentation for a preprocessed session (async)."""
    sdir = _find_session_dir(data_dir, session_id)
    if sdir is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    job_id = _new_job(session_id, "segment")
    background_tasks.add_task(_run_segment, job_id, sdir)
    return {"job_id": job_id, "status": "queued"}


@router.post("/chunk/{session_id}", status_code=202)
async def chunk_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    data_dir: str = Depends(get_data_dir),
) -> dict:
    """Trigger LeRobot chunking for a segmented session (async)."""
    sdir = _find_session_dir(data_dir, session_id)
    if sdir is None:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    job_id = _new_job(session_id, "chunk")
    background_tasks.add_task(_run_chunk, job_id, sdir)
    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}", response_model=JobStatus)
async def job_status(job_id: str) -> JobStatus:
    """Get the current status of a pipeline job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return JobStatus(**job)

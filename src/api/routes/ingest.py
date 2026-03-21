"""
Ingest router: server-side receiver for Raspberry Pi chunk uploads.

Endpoints
---------
POST   /api/ingest/chunk                      — receive a single 30-s chunk
POST   /api/ingest/finalize/{session_id}      — trigger assembly + pipeline
GET    /api/ingest/session/{session_id}/status — upload / pipeline progress
GET    /api/ingest/sessions                   — list all ingested sessions
DELETE /api/ingest/session/{session_id}        — delete raw chunk data
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Optional pipeline imports ─────────────────────────────────────────────────
# Each import is attempted independently so the API still starts even when
# optional dependencies (ffmpeg, torch, etc.) are not installed.

try:
    from ...preprocess.assembler import ChunkAssembler
    _HAS_ASSEMBLER = True
except ImportError as _e:
    logger.warning("ChunkAssembler unavailable: %s", _e)
    _HAS_ASSEMBLER = False
    ChunkAssembler = None  # type: ignore[assignment,misc]

try:
    from ...preprocess.gps import GPSPreprocessor
    _HAS_GPS = True
except ImportError as _e:
    logger.warning("GPSPreprocessor unavailable: %s", _e)
    _HAS_GPS = False
    GPSPreprocessor = None  # type: ignore[assignment,misc]

try:
    from ...preprocess.sync import StreamSynchronizer
    _HAS_SYNC = True
except ImportError as _e:
    logger.warning("StreamSynchronizer unavailable: %s", _e)
    _HAS_SYNC = False
    StreamSynchronizer = None  # type: ignore[assignment,misc]

try:
    from ...preprocess.hdf5_writer import HDF5Writer as VideoPreprocessor
    _HAS_VIDEO = True
except ImportError as _e:
    logger.warning("VideoPreprocessor unavailable: %s", _e)
    _HAS_VIDEO = False
    VideoPreprocessor = None  # type: ignore[assignment,misc]

# AudioPreprocessor is in the preprocess package (may be added later)
try:
    from ...preprocess import audio as _audio_module
    AudioPreprocessor = getattr(_audio_module, "AudioPreprocessor", None)
    _HAS_AUDIO = AudioPreprocessor is not None
except ImportError as _e:
    logger.warning("AudioPreprocessor unavailable: %s", _e)
    _HAS_AUDIO = False
    AudioPreprocessor = None  # type: ignore[assignment,misc]

# IMUPreprocessor (may be added later)
try:
    from ...preprocess import imu as _imu_module
    IMUPreprocessor = getattr(_imu_module, "IMUPreprocessor", None)
    _HAS_IMU = IMUPreprocessor is not None
except ImportError as _e:
    logger.warning("IMUPreprocessor unavailable: %s", _e)
    _HAS_IMU = False
    IMUPreprocessor = None  # type: ignore[assignment,misc]

# HindiTranscriber
try:
    from ...segmentation.transcriber import HindiTranscriber
    _HAS_TRANSCRIBER = True
except ImportError as _e:
    logger.warning("HindiTranscriber unavailable: %s", _e)
    _HAS_TRANSCRIBER = False
    HindiTranscriber = None  # type: ignore[assignment,misc]

# LeRobotChunker
try:
    from ...segmentation.lerobot_chunker import LeRobotChunker
    _HAS_CHUNKER = True
except ImportError as _e:
    logger.warning("LeRobotChunker unavailable: %s", _e)
    _HAS_CHUNKER = False
    LeRobotChunker = None  # type: ignore[assignment,misc]

# HDF5Store
try:
    from ...storage.hdf5_store import HDF5Store
    _HAS_HDF5STORE = True
except ImportError as _e:
    logger.warning("HDF5Store unavailable: %s", _e)
    _HAS_HDF5STORE = False
    HDF5Store = None  # type: ignore[assignment,misc]

# ── Job state ─────────────────────────────────────────────────────────────────

@dataclass
class JobState:
    """Tracks the status of an async processing job."""
    job_id:       str
    session_id:   str
    status:       str        # queued | running | done | error
    stage:        str        # assembly | sync | video | audio | imu | gps | transcribe | chunk | done
    progress_pct: float
    error:        Optional[str]
    created_at:   str
    updated_at:   str


# Module-level job registry (process-local; for production use Redis/DB)
JOBS: Dict[str, JobState] = {}

# Session registry: session_id → session info dict
SESSIONS: Dict[str, Dict[str, Any]] = {}

# ── Pydantic response models ──────────────────────────────────────────────────

class ChunkReceived(BaseModel):
    status:         str
    session_id:     str
    chunk_index:    int
    bytes_received: int


class FinalizeResponse(BaseModel):
    job_id:     str
    status:     str
    session_id: str


class SessionStatus(BaseModel):
    session_id:        str
    chunks_received:   int
    last_chunk_index:  Optional[int]
    is_finalized:      bool
    assembly_status:   str
    pipeline_status:   str


class SessionListItem(BaseModel):
    session_id:      str
    chunks_received: int
    is_finalized:    bool
    last_seen:       str


# ── Helpers ───────────────────────────────────────────────────────────────────

_SESSION_ID_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9\-_]{1,127}$"
)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _validate_session_id(session_id: str) -> None:
    """Raise HTTPException(422) if session_id has an invalid format."""
    if not (_UUID_RE.match(session_id) or _SESSION_ID_RE.match(session_id)):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid session_id format: '{session_id}'. "
                   "Must be a UUID or an alphanumeric slug (max 128 chars).",
        )


def _get_raw_dir(request: Request) -> Path:
    """Resolve the data directory from app state or environment."""
    data_dir = getattr(request.app.state, "data_dir", None) or os.environ.get(
        "DATA_DIR", "/data/sessions"
    )
    return Path(data_dir)


def _get_or_create_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "session_id":       session_id,
            "chunks_received":  0,
            "last_chunk_index": None,
            "is_finalized":     False,
            "assembly_status":  "none",
            "pipeline_status":  "none",
            "last_seen":        datetime.utcnow().isoformat(),
        }
    return SESSIONS[session_id]


def _save_upload_file(upload_file: UploadFile, dest: Path) -> int:
    """Write an UploadFile to disk and return byte count."""
    content = upload_file.file.read()
    dest.write_bytes(content)
    return len(content)


# ── Background pipeline ───────────────────────────────────────────────────────

def _run_pipeline(job_id: str, session_id: str, session_dir: Path) -> None:
    """
    Execute the full post-collection pipeline in a background thread.

    Stages (skipped gracefully if the required module is unavailable):
        1. assembly   — concatenate chunks
        2. sync       — align streams
        3. gps        — GPS preprocessing
        4. transcribe — Hindi ASR
        5. chunk      — LeRobot episode chunking
    """
    job = JOBS[job_id]

    def _update(stage: str, pct: float, status: str = "running") -> None:
        job.stage       = stage
        job.progress_pct = pct
        job.status      = status
        job.updated_at  = datetime.utcnow().isoformat()
        logger.info("[job:%s] stage=%s  progress=%.0f%%", job_id, stage, pct)

    try:
        assembled_session = None

        # ── Stage 1: Assembly ────────────────────────────────────────────────
        _update("assembly", 5.0)
        if _HAS_ASSEMBLER:
            assembler = ChunkAssembler()
            try:
                assembled_session = assembler.assemble_session(str(session_dir))
                SESSIONS[session_id]["assembly_status"] = "done"
                logger.info("Assembly complete for session %s", session_id)
            except Exception as exc:
                logger.error("Assembly failed: %s", exc, exc_info=True)
                SESSIONS[session_id]["assembly_status"] = "error"
        else:
            logger.warning("Assembly skipped (ChunkAssembler unavailable)")
            SESSIONS[session_id]["assembly_status"] = "skipped"

        # ── Stage 2: Stream sync ──────────────────────────────────────────────
        _update("sync", 20.0)
        synced_data = None
        if _HAS_SYNC and assembled_session:
            assembled_dir = Path(assembled_session.video_path).parent
            video_path = assembled_session.video_path
            audio_path = assembled_session.audio_path
            imu_path   = assembled_session.imu_path
            meta_path  = str(assembled_dir / "session_meta.json")
            try:
                syncer = StreamSynchronizer()
                synced_data = syncer.align_streams(
                    video_path=video_path,
                    audio_path=audio_path,
                    imu_csv_path=imu_path,
                    metadata_path=meta_path,
                )
            except Exception as exc:
                logger.error("Sync failed: %s", exc, exc_info=True)
        else:
            logger.warning("Sync skipped (StreamSynchronizer unavailable or no assembly)")

        # ── Stage 3: GPS preprocessing ────────────────────────────────────────
        _update("gps", 40.0)
        if _HAS_GPS and assembled_session and Path(assembled_session.gps_path).exists():
            gps_preprocessor = GPSPreprocessor()
            try:
                gps_df = gps_preprocessor.load_gps(assembled_session.gps_path)
                gps_df = gps_preprocessor.kalman_smooth(gps_df)
                gps_df = gps_preprocessor.compute_derived(gps_df)
                out_path = str(session_dir / "assembled" / "gps_processed.parquet")
                gps_preprocessor.save_processed(gps_df, out_path)
                logger.info("GPS preprocessing done → %s", out_path)
            except Exception as exc:
                logger.error("GPS preprocessing failed: %s", exc, exc_info=True)
        else:
            logger.warning("GPS preprocessing skipped")

        # ── Stage 4: Transcription ────────────────────────────────────────────
        _update("transcribe", 60.0)
        if _HAS_TRANSCRIBER and assembled_session:
            try:
                transcriber = HindiTranscriber()
                result = transcriber.transcribe(assembled_session.audio_path)
                SESSIONS[session_id]["language_instruction"] = result.get("text", "")
                logger.info("Transcription done")
            except Exception as exc:
                logger.error("Transcription failed: %s", exc, exc_info=True)
        else:
            logger.warning("Transcription skipped (HindiTranscriber unavailable)")

        # ── Stage 5: Episode chunking ─────────────────────────────────────────
        _update("chunk", 80.0)
        if _HAS_CHUNKER and synced_data:
            try:
                chunker = LeRobotChunker()
                chunker.chunk(synced_data, output_dir=str(session_dir / "episodes"))
                logger.info("Episode chunking done")
            except Exception as exc:
                logger.error("Episode chunking failed: %s", exc, exc_info=True)
        else:
            logger.warning("Episode chunking skipped (LeRobotChunker unavailable or no sync data)")

        # ── Done ──────────────────────────────────────────────────────────────
        SESSIONS[session_id]["pipeline_status"] = "done"
        _update("done", 100.0, status="done")

    except Exception as exc:
        job.status    = "error"
        job.error     = str(exc)
        job.updated_at = datetime.utcnow().isoformat()
        SESSIONS[session_id]["pipeline_status"] = "error"
        logger.error("[job:%s] Pipeline error: %s", job_id, exc, exc_info=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chunk", response_model=ChunkReceived, status_code=200)
async def receive_chunk(
    request: Request,
    session_id:  str         = File(...),
    chunk_index: int         = File(...),
    device_id:   str         = File(...),
    video:       UploadFile  = File(...),
    audio:       UploadFile  = File(...),
    imu:         UploadFile  = File(...),
    gps:         UploadFile  = File(...),
    chunk_meta:  UploadFile  = File(...),
) -> ChunkReceived:
    """
    Receive a single 30-second data chunk from a Raspberry Pi headset.

    The multipart form must include the five files (video, audio, imu, gps,
    chunk_meta) plus form fields session_id, chunk_index, and device_id.
    """
    _validate_session_id(session_id)

    raw_dir    = _get_raw_dir(request)
    chunk_dir  = raw_dir / session_id / "chunks" / f"chunk_{chunk_index:04d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    file_map = {
        f"video_{chunk_index:04d}.mp4":      video,
        f"audio_{chunk_index:04d}.wav":      audio,
        f"imu_{chunk_index:04d}.csv":        imu,
        f"gps_{chunk_index:04d}.csv":        gps,
        f"chunk_meta_{chunk_index:04d}.json": chunk_meta,
    }

    for filename, upload in file_map.items():
        dest = chunk_dir / filename
        total_bytes += _save_upload_file(upload, dest)

    # Update session registry
    sess = _get_or_create_session(session_id)
    sess["chunks_received"] += 1
    sess["last_chunk_index"] = max(
        sess.get("last_chunk_index") or -1,
        chunk_index,
    )
    sess["last_seen"] = datetime.utcnow().isoformat()

    logger.info(
        "Chunk received: session=%s chunk=%d device=%s bytes=%d",
        session_id, chunk_index, device_id, total_bytes,
    )

    return ChunkReceived(
        status="received",
        session_id=session_id,
        chunk_index=chunk_index,
        bytes_received=total_bytes,
    )


@router.post("/finalize/{session_id}", response_model=FinalizeResponse, status_code=202)
async def finalize_session(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> FinalizeResponse:
    """
    Mark a session as complete and trigger the full preprocessing pipeline.

    The pipeline runs in a background thread; poll
    ``GET /api/ingest/session/{session_id}/status`` for progress.
    """
    _validate_session_id(session_id)

    raw_dir     = _get_raw_dir(request)
    session_dir = raw_dir / session_id

    if not session_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No data found for session '{session_id}'.",
        )

    sess = _get_or_create_session(session_id)
    sess["is_finalized"]    = True
    sess["assembly_status"] = "queued"
    sess["pipeline_status"] = "queued"

    job_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()
    JOBS[job_id] = JobState(
        job_id=job_id,
        session_id=session_id,
        status="queued",
        stage="queued",
        progress_pct=0.0,
        error=None,
        created_at=now,
        updated_at=now,
    )

    background_tasks.add_task(_run_pipeline, job_id, session_id, session_dir)
    logger.info("Pipeline job %s queued for session %s", job_id, session_id)

    return FinalizeResponse(job_id=job_id, status="queued", session_id=session_id)


@router.get("/session/{session_id}/status", response_model=SessionStatus)
async def get_session_status(session_id: str) -> SessionStatus:
    """Return current upload and pipeline status for a session."""
    _validate_session_id(session_id)

    sess = SESSIONS.get(session_id)
    if not sess:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    return SessionStatus(
        session_id=session_id,
        chunks_received=sess.get("chunks_received", 0),
        last_chunk_index=sess.get("last_chunk_index"),
        is_finalized=bool(sess.get("is_finalized", False)),
        assembly_status=str(sess.get("assembly_status", "none")),
        pipeline_status=str(sess.get("pipeline_status", "none")),
    )


@router.get("/sessions", response_model=List[SessionListItem])
async def list_ingest_sessions() -> List[SessionListItem]:
    """List all sessions that have sent at least one chunk."""
    result: List[SessionListItem] = []
    for sid, sess in SESSIONS.items():
        result.append(
            SessionListItem(
                session_id=sid,
                chunks_received=sess.get("chunks_received", 0),
                is_finalized=bool(sess.get("is_finalized", False)),
                last_seen=sess.get("last_seen", ""),
            )
        )
    return sorted(result, key=lambda x: x.last_seen, reverse=True)


@router.delete("/session/{session_id}", status_code=200)
async def delete_ingest_session(
    session_id: str,
    request: Request,
) -> dict:
    """
    Delete the raw chunk files for a session after processing is confirmed.

    Does not remove assembled or episode data — only the raw ``chunks/``
    sub-directory is deleted.
    """
    _validate_session_id(session_id)

    raw_dir     = _get_raw_dir(request)
    session_dir = raw_dir / session_id
    chunks_dir  = session_dir / "chunks"

    if not session_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Session directory not found for '{session_id}'.",
        )

    deleted_path = None
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
        deleted_path = str(chunks_dir)
        logger.info("Deleted raw chunks for session %s: %s", session_id, chunks_dir)

    # Remove from in-memory registry
    SESSIONS.pop(session_id, None)

    return {
        "status":       "deleted",
        "session_id":   session_id,
        "deleted_path": deleted_path,
    }

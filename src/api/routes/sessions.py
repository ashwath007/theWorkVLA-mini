"""
Sessions router: CRUD operations for recorded headset sessions.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    session_id: str
    data_dir: str
    video_frames: int
    duration_sec: float
    start_time: Optional[float]
    has_video: bool
    has_audio: bool
    has_imu: bool


class SessionDetail(SessionSummary):
    metadata: Dict[str, Any]
    episode_count: int


# ── Dependency ────────────────────────────────────────────────────────────────

def get_data_dir(request: Request) -> str:
    return getattr(request.app.state, "data_dir", os.environ.get("DATA_DIR", "/data/sessions"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_all_sessions(data_dir: str) -> List[Dict[str, Any]]:
    """Scan data_dir for metadata.json files and return parsed list."""
    base     = Path(data_dir)
    sessions = []
    for meta_path in sorted(base.rglob("metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
            meta["data_dir"] = str(meta_path.parent)
            sessions.append(meta)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", meta_path, exc)
    return sessions


def _get_session_by_id(data_dir: str, session_id: str) -> Optional[Dict[str, Any]]:
    """Find a session by session_id."""
    for s in _find_all_sessions(data_dir):
        if s.get("session_id") == session_id:
            return s
    return None


def _session_to_summary(meta: Dict[str, Any]) -> SessionSummary:
    sdir   = Path(meta.get("data_dir", ""))
    dur    = (meta.get("end_time") or 0) - (meta.get("start_time") or 0)
    return SessionSummary(
        session_id=meta.get("session_id", "?"),
        data_dir=str(sdir),
        video_frames=meta.get("video_frames", 0),
        duration_sec=round(dur, 2),
        start_time=meta.get("start_time"),
        has_video=(sdir / "video.mp4").exists(),
        has_audio=(sdir / "audio.wav").exists(),
        has_imu=(sdir / "imu.csv").exists(),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=List[SessionSummary])
async def list_sessions(data_dir: str = Depends(get_data_dir)) -> List[SessionSummary]:
    """List all recorded sessions with summary stats."""
    metas = _find_all_sessions(data_dir)
    return [_session_to_summary(m) for m in metas]


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    data_dir: str = Depends(get_data_dir),
) -> SessionDetail:
    """Return full metadata and episode count for a session."""
    meta = _get_session_by_id(data_dir, session_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    sdir          = Path(meta.get("data_dir", ""))
    episode_count = len(list(sdir.rglob("*.hdf5")))
    summary       = _session_to_summary(meta)

    return SessionDetail(
        **summary.dict(),
        metadata=meta,
        episode_count=episode_count,
    )


@router.post("/upload", status_code=202)
async def upload_session(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    data_dir: str = Depends(get_data_dir),
) -> dict:
    """
    Upload a ZIP archive containing a raw session
    (video.mp4, audio.wav, imu.csv, metadata.json).

    The ZIP is extracted asynchronously.
    """
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archives are accepted.")

    session_id  = str(uuid.uuid4())
    session_dir = Path(data_dir) / "uploads" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    zip_path = session_dir / "upload.zip"
    contents = await file.read()
    zip_path.write_bytes(contents)

    def _extract(zip_p: Path, dest: Path) -> None:
        try:
            with zipfile.ZipFile(zip_p, "r") as zf:
                zf.extractall(dest)
            zip_p.unlink()
            logger.info("Session %s extracted to %s", session_id, dest)
        except Exception as exc:
            logger.error("Extraction failed for %s: %s", session_id, exc)

    background_tasks.add_task(_extract, zip_path, session_dir)

    return {
        "status":     "accepted",
        "session_id": session_id,
        "message":    "Archive is being extracted in the background.",
    }


@router.delete("/{session_id}", status_code=200)
async def delete_session(
    session_id: str,
    data_dir: str = Depends(get_data_dir),
) -> dict:
    """Delete all data for the given session_id."""
    meta = _get_session_by_id(data_dir, session_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    sdir = Path(meta.get("data_dir", ""))
    if sdir.exists() and sdir.is_dir():
        shutil.rmtree(sdir)
        logger.info("Deleted session directory: %s", sdir)
        return {"status": "deleted", "session_id": session_id, "path": str(sdir)}
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Session directory does not exist: {sdir}",
        )

"""
Labeling router: LabelStudio integration endpoints.

POST /labeling/push/{session_id}      — push segment frames to LabelStudio
GET  /labeling/pull/{session_id}      — pull annotations from LabelStudio
POST /labeling/export/{session_id}    — apply annotations → labeled HDF5 episodes
GET  /labeling/auto/{session_id}      — get auto-label preview (no push)
GET  /labeling/stats/{project_id}     — annotation stats for a project
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Lazy imports ──────────────────────────────────────────────────────────────

def _get_ls_client():
    from ...labeling.labelstudio import LabelStudioClient
    url = os.environ.get("LABELSTUDIO_URL", "http://localhost:8080")
    key = os.environ.get("LABELSTUDIO_API_KEY", "")
    if not key:
        raise HTTPException(status_code=503, detail="LABELSTUDIO_API_KEY not set")
    return LabelStudioClient(url, key)


def _get_episodes_dir(session_id: str) -> Path:
    data_dir = os.environ.get("DATA_DIR", "/data/sessions")
    ep_dir   = Path(data_dir) / session_id / "episodes"
    if not ep_dir.exists():
        raise HTTPException(status_code=404, detail=f"No episodes found for session '{session_id}'")
    return ep_dir


def _get_frames_dir(session_id: str) -> Path:
    data_dir = os.environ.get("DATA_DIR", "/data/sessions")
    return Path(data_dir) / session_id


# ── Request / Response models ─────────────────────────────────────────────────

class PushRequest(BaseModel):
    project_title: str = "india-vla"
    max_frames:    int = 200      # limit frames pushed per session
    use_segments:  bool = True    # push one frame per segment (vs every N frames)
    auto_label:    bool = True    # attach auto-label predictions


class ExportRequest(BaseModel):
    project_id: int
    output_subdir: str = "labeled_episodes"
    overwrite:     bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/push/{session_id}")
async def push_to_labelstudio(
    session_id: str,
    req: PushRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Push session frames (or segment key-frames) to LabelStudio for labeling.
    Optionally attaches auto-label predictions to save annotator time.
    """
    data_dir    = os.environ.get("DATA_DIR", "/data/sessions")
    session_dir = Path(data_dir) / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    background_tasks.add_task(
        _push_task,
        session_id=session_id,
        session_dir=session_dir,
        project_title=req.project_title,
        max_frames=req.max_frames,
        use_segments=req.use_segments,
        auto_label=req.auto_label,
    )
    return {"status": "queued", "session_id": session_id, "project_title": req.project_title}


@router.get("/pull/{session_id}")
async def pull_from_labelstudio(
    session_id: str,
    project_id: int,
) -> Dict[str, Any]:
    """Pull completed annotations for a session from LabelStudio."""
    client      = _get_ls_client()
    annotations = client.pull_annotations(project_id)

    # Filter to this session
    session_anns = [a for a in annotations if a.get("session_id") == session_id]

    # Save to disk
    data_dir = os.environ.get("DATA_DIR", "/data/sessions")
    out_path = Path(data_dir) / session_id / "annotations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(session_anns, ensure_ascii=False, indent=2))

    return {
        "session_id":       session_id,
        "project_id":       project_id,
        "total_annotations": len(session_anns),
        "saved_to":         str(out_path),
    }


@router.post("/export/{session_id}")
async def export_labels(
    session_id: str,
    req: ExportRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Pull annotations from LabelStudio and apply them to HDF5 episodes.
    Creates labeled_episodes/ alongside episodes/.
    """
    data_dir    = os.environ.get("DATA_DIR", "/data/sessions")
    session_dir = Path(data_dir) / session_id
    ann_path    = session_dir / "annotations.json"

    if not ann_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No annotations file found. Run /labeling/pull/{session_id} first."
        )

    background_tasks.add_task(
        _export_task,
        session_id=session_id,
        session_dir=session_dir,
        ann_path=ann_path,
        output_subdir=req.output_subdir,
        overwrite=req.overwrite,
    )
    return {"status": "queued", "session_id": session_id}


@router.get("/auto/{session_id}")
async def get_auto_labels(session_id: str) -> Dict[str, Any]:
    """
    Return auto-generated label suggestions for a session (preview, no push).
    Reads transcription + segments from assembled/ directory.
    """
    data_dir    = os.environ.get("DATA_DIR", "/data/sessions")
    session_dir = Path(data_dir) / session_id
    assembled   = session_dir / "assembled"

    segments_path    = assembled / "segments.json"
    transcription_path = assembled / "transcription.json"

    if not segments_path.exists():
        raise HTTPException(status_code=404, detail="Segments not found. Run pipeline first.")

    segments     = json.loads(segments_path.read_text())
    transcription = {}
    if transcription_path.exists():
        transcription = json.loads(transcription_path.read_text())

    try:
        from ...labeling.auto_labeler import AutoLabeler
        labeler = AutoLabeler()
        labels  = labeler.label_segments(
            segments=segments,
            frames=[],           # no frames for preview — text-only
            transcription=transcription,
        )
        summary = labeler.summary(labels)
        return {
            "session_id": session_id,
            "num_segments": len(segments),
            "auto_labels": [
                {
                    "segment_index":  lb.segment_index,
                    "action_type":    lb.action_type,
                    "scenario":       lb.scenario,
                    "instruction_hi": lb.instruction_hi,
                    "instruction_en": lb.instruction_en,
                    "confidence":     lb.confidence,
                    "motion_level":   lb.motion_level,
                    "sources":        lb.sources,
                }
                for lb in labels
            ],
            "summary": summary,
        }
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"AutoLabeler unavailable: {exc}")


@router.get("/stats/{project_id}")
async def get_labeling_stats(project_id: int) -> Dict[str, Any]:
    """Return annotation statistics for a LabelStudio project."""
    client  = _get_ls_client()
    project = client.get_project(project_id)
    tasks   = client.get_tasks(project_id, page_size=1000)

    total_tasks       = len(tasks)
    annotated_tasks   = sum(1 for t in tasks if t.get("annotations"))
    total_annotations = sum(len(t.get("annotations", [])) for t in tasks)

    return {
        "project_id":        project_id,
        "project_title":     project.get("title", ""),
        "total_tasks":       total_tasks,
        "annotated_tasks":   annotated_tasks,
        "pending_tasks":     total_tasks - annotated_tasks,
        "total_annotations": total_annotations,
        "completion_pct":    round(annotated_tasks / total_tasks * 100, 1) if total_tasks else 0,
    }


# ── Background task implementations ──────────────────────────────────────────

def _push_task(
    session_id: str,
    session_dir: Path,
    project_title: str,
    max_frames: int,
    use_segments: bool,
    auto_label: bool,
) -> None:
    try:
        import numpy as np
        from ...labeling.labelstudio import LabelStudioClient
        from ...labeling.auto_labeler import AutoLabeler

        url = os.environ.get("LABELSTUDIO_URL", "http://localhost:8080")
        key = os.environ.get("LABELSTUDIO_API_KEY", "")
        if not key:
            logger.error("LABELSTUDIO_API_KEY not set — cannot push tasks")
            return

        client     = LabelStudioClient(url, key)
        project_id = client.get_or_create_project(project_title)

        assembled = session_dir / "assembled"
        segments_path = assembled / "segments.json"
        transcription_path = assembled / "transcription.json"

        segments     = json.loads(segments_path.read_text()) if segments_path.exists() else []
        transcription = json.loads(transcription_path.read_text()) if transcription_path.exists() else {}

        # Load representative frames from assembled video
        frames = _load_key_frames(session_dir, segments, max_frames)

        auto_labels_ls = []
        if auto_label and segments:
            labeler = AutoLabeler(default_scenario=session_dir.name.split("_")[0] if "_" in session_dir.name else "unknown")
            labels  = labeler.label_segments(
                segments=segments,
                frames=frames,
                transcription=transcription,
            )
            auto_labels_ls = labeler.to_labelstudio_format(labels)

        if use_segments and segments:
            task_ids = client.push_segment_tasks(
                project_id=project_id,
                frames=frames,
                segments=segments,
                session_id=session_id,
                auto_labels=auto_labels_ls or None,
            )
        else:
            task_ids = client.push_frames(
                project_id=project_id,
                frames=frames[:max_frames],
                session_id=session_id,
            )

        logger.info("Pushed %d tasks to project %d for session %s", len(task_ids), project_id, session_id)

        # Save project_id mapping
        mapping_path = session_dir / "labelstudio_project.json"
        mapping_path.write_text(json.dumps({"project_id": project_id, "task_ids": task_ids}))

    except Exception as exc:
        logger.error("Push task failed for session %s: %s", session_id, exc, exc_info=True)


def _export_task(
    session_id: str,
    session_dir: Path,
    ann_path: Path,
    output_subdir: str,
    overwrite: bool,
) -> None:
    try:
        from ...labeling.exporter import LabelExporter
        episodes_dir = session_dir / "episodes"
        output_dir   = session_dir / output_subdir

        exporter = LabelExporter(str(episodes_dir))
        result   = exporter.export_from_file(
            str(ann_path),
            output_dir=output_dir,
            overwrite=overwrite,
        )
        logger.info("Export done for session %s: %s", session_id, result)
    except Exception as exc:
        logger.error("Export task failed for session %s: %s", session_id, exc, exc_info=True)


def _load_key_frames(
    session_dir: Path,
    segments: List[Dict],
    max_frames: int,
) -> List:
    """Load one representative frame per segment from assembled video."""
    try:
        import cv2
        import numpy as np
        video_path = session_dir / "assembled" / "video.mp4"
        if not video_path.exists():
            return []

        cap    = cv2.VideoCapture(str(video_path))
        frames = []

        target_indices = []
        if segments:
            for seg in segments[:max_frames]:
                mid = (seg["start_frame"] + seg["end_frame"]) // 2
                target_indices.append(mid)
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step  = max(1, total // max_frames)
            target_indices = list(range(0, total, step))[:max_frames]

        for fidx in sorted(target_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb.astype(np.uint8))

        cap.release()
        return frames
    except Exception as exc:
        logger.warning("Could not load key frames: %s", exc)
        return []

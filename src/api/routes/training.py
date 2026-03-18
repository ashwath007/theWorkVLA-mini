"""
Training router: start training jobs, check progress, list and export checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Job store (in-memory) ─────────────────────────────────────────────────────

_training_jobs: Dict[str, Dict[str, Any]] = {}


def _new_training_job(config_dict: dict) -> str:
    job_id = str(uuid.uuid4())
    _training_jobs[job_id] = {
        "job_id":     job_id,
        "status":     "queued",
        "message":    "Waiting to start …",
        "epoch":      0,
        "total_epochs": config_dict.get("epochs", 0),
        "train_loss": None,
        "val_loss":   None,
        "config":     config_dict,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    if job_id in _training_jobs:
        _training_jobs[job_id].update(kwargs, updated_at=time.time())


# ── Schemas ───────────────────────────────────────────────────────────────────

class TrainingRequest(BaseModel):
    dataset_dir:       str
    epochs:            int = 50
    batch_size:        int = 32
    learning_rate:     float = 3e-4
    hidden_dim:        int = 512
    language_model:    str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    freeze_language:   bool = True
    freeze_vision:     bool = False
    image_size:        int = 224
    max_seq_len:       int = 50
    grad_clip:         float = 1.0
    weight_decay:      float = 0.01
    warmup_steps:      int = 500
    save_every_n_epochs: int = 5
    output_dir:        Optional[str] = None


class TrainingStatus(BaseModel):
    job_id:       str
    status:       str
    message:      str
    epoch:        int
    total_epochs: int
    train_loss:   Optional[float]
    val_loss:     Optional[float]
    created_at:   float
    updated_at:   float


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    path:          str
    epoch:         int
    val_loss:      Optional[float]
    created_at:    float


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_model_dir(request: Request) -> str:
    return getattr(request.app.state, "model_dir", os.environ.get("MODEL_OUTPUT_DIR", "/models"))


# ── Background worker ─────────────────────────────────────────────────────────

def _run_training(job_id: str, req: TrainingRequest, output_dir: str) -> None:
    """Execute training in a background thread."""
    try:
        _update_job(job_id, status="running", message="Initializing …")

        from src.training.train import TrainingConfig, Trainer

        cfg = TrainingConfig(
            dataset_dir=req.dataset_dir,
            output_dir=output_dir,
            epochs=req.epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            hidden_dim=req.hidden_dim,
            language_model=req.language_model,
            freeze_language=req.freeze_language,
            freeze_vision=req.freeze_vision,
            image_size=req.image_size,
            max_seq_len=req.max_seq_len,
            grad_clip=req.grad_clip,
            weight_decay=req.weight_decay,
            warmup_steps=req.warmup_steps,
            save_every_n_epochs=req.save_every_n_epochs,
        )

        # Patch trainer to report progress via job store
        import torch.nn as nn
        from src.training.train import Trainer as _Trainer

        class PatchedTrainer(_Trainer):
            def _train_epoch(self, model, loader, optimizer, scheduler, loss_fn, epoch, progress):
                loss = super()._train_epoch(model, loader, optimizer, scheduler, loss_fn, epoch, progress)
                _update_job(
                    job_id,
                    epoch=epoch,
                    train_loss=round(loss, 4),
                    message=f"Epoch {epoch}/{cfg.epochs} train_loss={loss:.4f}",
                )
                return loss

            def _validate(self, model, loader, loss_fn, epoch):
                loss = super()._validate(model, loader, loss_fn, epoch)
                _update_job(job_id, val_loss=round(loss, 4))
                return loss

        trainer = PatchedTrainer(cfg)
        trainer.train()

        _update_job(job_id, status="completed", message="Training finished.")
        logger.info("Training job %s completed.", job_id)

    except Exception as exc:
        _update_job(job_id, status="failed", message=str(exc))
        logger.exception("Training job %s failed.", job_id)


def _run_export(job_id_or_path: str, export_dir: str) -> None:
    """Export a checkpoint to Hugging Face format."""
    try:
        from src.training.model import IndiaVLAModel
        model = IndiaVLAModel.from_pretrained(job_id_or_path)
        model.save_pretrained(export_dir)
        logger.info("Exported checkpoint to %s", export_dir)
    except Exception as exc:
        logger.exception("Export failed for %s: %s", job_id_or_path, exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start", status_code=202)
async def start_training(
    req: TrainingRequest,
    background_tasks: BackgroundTasks,
    model_dir: str = Depends(get_model_dir),
) -> dict:
    """Start a training job with the given config."""
    output_dir = req.output_dir or str(Path(model_dir) / "training" / str(uuid.uuid4())[:8])
    job_id     = _new_training_job(req.dict())
    background_tasks.add_task(_run_training, job_id, req, output_dir)
    return {"job_id": job_id, "status": "queued", "output_dir": output_dir}


@router.get("/status/{job_id}", response_model=TrainingStatus)
async def training_status(job_id: str) -> TrainingStatus:
    """Get the current status/progress of a training job."""
    job = _training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Training job '{job_id}' not found.")
    return TrainingStatus(
        job_id=job["job_id"],
        status=job["status"],
        message=job["message"],
        epoch=job["epoch"],
        total_epochs=job["total_epochs"],
        train_loss=job.get("train_loss"),
        val_loss=job.get("val_loss"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.get("/checkpoints", response_model=List[CheckpointInfo])
async def list_checkpoints(
    model_dir: str = Depends(get_model_dir),
) -> List[CheckpointInfo]:
    """List all saved model checkpoints."""
    base  = Path(model_dir)
    ckpts = []

    for p in sorted(base.rglob("model.pt")):
        ckpt_dir = p.parent
        config_path = ckpt_dir / "config.json"
        opt_path    = ckpt_dir / "optimizer.pt"

        val_loss    = None
        epoch       = 0
        created_at  = p.stat().st_mtime

        if opt_path.exists():
            try:
                import torch
                opt_data = torch.load(str(opt_path), map_location="cpu")
                val_loss = opt_data.get("val_loss")
                epoch    = opt_data.get("epoch", 0)
            except Exception:
                pass

        ckpts.append(CheckpointInfo(
            checkpoint_id=ckpt_dir.name,
            path=str(ckpt_dir),
            epoch=epoch,
            val_loss=val_loss,
            created_at=created_at,
        ))

    return sorted(ckpts, key=lambda c: c.epoch)


@router.post("/export/{checkpoint_id}", status_code=202)
async def export_checkpoint(
    checkpoint_id: str,
    background_tasks: BackgroundTasks,
    model_dir: str = Depends(get_model_dir),
) -> dict:
    """Export a checkpoint to a clean HF-compatible format."""
    # Search for checkpoint directory
    ckpt_dir = None
    for p in Path(model_dir).rglob("model.pt"):
        if p.parent.name == checkpoint_id:
            ckpt_dir = p.parent
            break

    if ckpt_dir is None:
        raise HTTPException(404, f"Checkpoint '{checkpoint_id}' not found.")

    export_path = str(Path(model_dir) / "exports" / checkpoint_id)
    background_tasks.add_task(_run_export, str(ckpt_dir), export_path)
    return {"status": "accepted", "checkpoint_id": checkpoint_id, "export_path": export_path}

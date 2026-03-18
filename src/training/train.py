"""
Training script for IndiaVLAModel.

Behaviour cloning (MSE loss on continuous actions) with:
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - TensorBoard logging
  - Periodic checkpoint saving

Usage
-----
    python -m src.training.train \\
        --dataset-dir /data/episodes \\
        --output-dir ./checkpoints \\
        --epochs 50
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

from .dataset import IndiaVLADataset
from .model import IndiaVLAModel

logger  = logging.getLogger(__name__)
console = Console()
app     = typer.Typer(name="train", add_completion=False)


@dataclass
class TrainingConfig:
    """Full training configuration with sensible defaults."""
    # Data
    dataset_dir:       str   = "/data/episodes"
    output_dir:        str   = "./checkpoints"
    log_dir:           str   = "./logs"
    train_split:       float = 0.8
    # Model
    hidden_dim:        int   = 512
    action_dim:        int   = 7
    num_heads:         int   = 8
    dropout:           float = 0.1
    image_size:        int   = 224
    max_seq_len:       int   = 50
    language_model:    str   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    freeze_language:   bool  = True
    freeze_vision:     bool  = False
    # Training
    epochs:            int   = 50
    batch_size:        int   = 32
    learning_rate:     float = 3e-4
    weight_decay:      float = 0.01
    warmup_steps:      int   = 500
    grad_clip:         float = 1.0
    num_workers:       int   = 4
    save_every_n_epochs: int = 5
    # Misc
    seed:              int   = 42
    device:            str   = "auto"   # 'auto', 'cpu', 'cuda', 'mps'

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        import yaml
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        flat: Dict = {}
        for section in ("training", "model", "data", "output"):
            flat.update(raw.get(section, {}))
        # Map YAML keys to dataclass field names
        mapping = {
            "batch_size":            "batch_size",
            "learning_rate":         "learning_rate",
            "epochs":                "epochs",
            "warmup_steps":          "warmup_steps",
            "grad_clip":             "grad_clip",
            "weight_decay":          "weight_decay",
            "vision_backbone":       None,   # handled separately
            "language_model":        "language_model",
            "action_dim":            "action_dim",
            "hidden_dim":            "hidden_dim",
            "num_heads":             "num_heads",
            "dropout":               "dropout",
            "image_size":            "image_size",
            "max_seq_len":           "max_seq_len",
            "train_split":           "train_split",
            "checkpoint_dir":        "output_dir",
            "log_dir":               "log_dir",
            "save_every_n_epochs":   "save_every_n_epochs",
        }
        kwargs = {}
        for yaml_key, dc_key in mapping.items():
            if dc_key and yaml_key in flat:
                kwargs[dc_key] = flat[yaml_key]
        return cls(**kwargs)


class Trainer:
    """
    Manages the full training and evaluation loop.

    Parameters
    ----------
    config : TrainingConfig
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = config.resolve_device()

        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        # Directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=config.log_dir)
        self._global_step = 0
        self._best_val_loss = float("inf")

    # ── Build components ─────────────────────────────────────────────────────────

    def _build_dataloaders(self) -> tuple:
        train_ds = IndiaVLADataset(
            data_dir=self.config.dataset_dir,
            split="train",
            image_size=self.config.image_size,
            max_seq_len=self.config.max_seq_len,
            train_split=self.config.train_split,
            tokenizer_name=self.config.language_model,
        )
        val_ds = IndiaVLADataset(
            data_dir=self.config.dataset_dir,
            split="validation",
            image_size=self.config.image_size,
            max_seq_len=self.config.max_seq_len,
            train_split=self.config.train_split,
            tokenizer_name=self.config.language_model,
        )
        pin  = self.device.type == "cuda"
        n_w  = self.config.num_workers

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size,
            shuffle=True, num_workers=n_w, pin_memory=pin, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size,
            shuffle=False, num_workers=n_w, pin_memory=pin,
        )
        return train_loader, val_loader

    def _build_model(self) -> IndiaVLAModel:
        model = IndiaVLAModel(
            hidden_dim=self.config.hidden_dim,
            action_dim=self.config.action_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            language_model_name=self.config.language_model,
            freeze_language=self.config.freeze_language,
            freeze_vision=self.config.freeze_vision,
        )
        return model.to(self.device)

    # ── Training loop ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Full training loop."""
        train_loader, val_loader = self._build_dataloaders()
        model   = self._build_model()
        loss_fn = nn.MSELoss()

        # Log parameter counts
        param_counts = model.count_parameters()
        total_params = sum(param_counts.values())
        logger.info("Trainable parameters: %d", total_params)
        for name, count in param_counts.items():
            logger.info("  %-12s  %d", name, count)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps   = self.config.epochs * max(1, len(train_loader))
        lr_scheduler  = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        self.config.save(os.path.join(self.config.output_dir, "training_config.json"))

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            epoch_task = progress.add_task("[cyan]Epochs", total=self.config.epochs)

            for epoch in range(1, self.config.epochs + 1):
                train_loss = self._train_epoch(
                    model, train_loader, optimizer, lr_scheduler, loss_fn, epoch, progress
                )
                val_loss = self._validate(model, val_loader, loss_fn, epoch)

                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val",   val_loss,   epoch)
                self.writer.add_scalar(
                    "LR", optimizer.param_groups[0]["lr"], epoch
                )

                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                    epoch, self.config.epochs,
                    train_loss, val_loss,
                    optimizer.param_groups[0]["lr"],
                )

                # Save periodic checkpoint
                if epoch % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(model, optimizer, epoch, val_loss)

                # Save best model
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    model.save_pretrained(
                        os.path.join(self.config.output_dir, "best_model")
                    )
                    logger.info("New best model saved (val_loss=%.4f)", val_loss)

                progress.advance(epoch_task)

        self.writer.close()
        console.print(f"[green]Training complete. Best val loss: {self._best_val_loss:.4f}[/green]")

    def _train_epoch(
        self,
        model: IndiaVLAModel,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler: CosineAnnealingLR,
        loss_fn: nn.MSELoss,
        epoch: int,
        progress,
    ) -> float:
        """Single training epoch; returns mean loss."""
        model.train()
        total_loss = 0.0
        n_batches  = 0

        batch_task = progress.add_task(
            f"[yellow]Epoch {epoch} train", total=len(loader)
        )

        for batch in loader:
            images      = batch["images"].to(self.device)
            lang_tokens = batch["language_tokens"].to(self.device)
            attn_mask   = batch["attention_mask"].to(self.device)
            imu         = batch["imu"].to(self.device)
            targets     = batch["actions"].to(self.device)

            optimizer.zero_grad()
            preds = model(images, lang_tokens, imu, attn_mask)
            loss  = loss_fn(preds, targets)
            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

            optimizer.step()

            # Warmup: use linear schedule for first warmup_steps, then cosine
            if self._global_step < self.config.warmup_steps:
                lr_scale = (self._global_step + 1) / max(1, self.config.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = self.config.learning_rate * lr_scale
            else:
                scheduler.step()

            total_loss += loss.item()
            n_batches  += 1
            self._global_step += 1

            self.writer.add_scalar("Loss/step", loss.item(), self._global_step)
            progress.advance(batch_task)

        progress.remove_task(batch_task)
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def _validate(
        self,
        model: IndiaVLAModel,
        loader: DataLoader,
        loss_fn: nn.MSELoss,
        epoch: int,
    ) -> float:
        """Validation loop; returns mean loss."""
        model.eval()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            images      = batch["images"].to(self.device)
            lang_tokens = batch["language_tokens"].to(self.device)
            attn_mask   = batch["attention_mask"].to(self.device)
            imu         = batch["imu"].to(self.device)
            targets     = batch["actions"].to(self.device)

            preds      = model(images, lang_tokens, imu, attn_mask)
            total_loss += loss_fn(preds, targets).item()
            n_batches  += 1

        return total_loss / max(1, n_batches)

    def _save_checkpoint(
        self,
        model: IndiaVLAModel,
        optimizer: AdamW,
        epoch: int,
        val_loss: float,
    ) -> None:
        ckpt_dir = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(ckpt_dir))
        torch.save(
            {
                "epoch":      epoch,
                "val_loss":   val_loss,
                "global_step": self._global_step,
                "optimizer":  optimizer.state_dict(),
            },
            ckpt_dir / "optimizer.pt",
        )
        logger.info("Checkpoint saved at epoch %d → %s", epoch, ckpt_dir)


# ── Entry point ─────────────────────────────────────────────────────────────────

def train(config: TrainingConfig) -> None:
    """Convenience wrapper for programmatic use."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    trainer = Trainer(config)
    trainer.train()


@app.command()
def main(
    dataset_dir:   str   = typer.Option("/data/episodes", "--dataset-dir"),
    output_dir:    str   = typer.Option("./checkpoints",  "--output-dir"),
    epochs:        int   = typer.Option(50,               "--epochs"),
    batch_size:    int   = typer.Option(32,               "--batch-size"),
    lr:            float = typer.Option(3e-4,             "--lr"),
    config_yaml:   Optional[str] = typer.Option(None,    "--config"),
) -> None:
    """Start VLA model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if config_yaml:
        cfg = TrainingConfig.from_yaml(config_yaml)
    else:
        cfg = TrainingConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
        )

    train(cfg)


if __name__ == "__main__":
    app()

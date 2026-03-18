"""
IndiaVLADataset: PyTorch Dataset for egocentric VLA training.

Loads LeRobot-format HDF5 episode files, applies data augmentation,
and tokenizes language instructions.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def _build_image_augmentation(image_size: int, is_train: bool) -> transforms.Compose:
    """Return a torchvision transform pipeline for images."""
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),                        # (C, H, W) float [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


class IndiaVLADataset(Dataset):
    """
    PyTorch Dataset for the India Egocentric VLA task.

    Each item returned is one frame from one episode.

    Parameters
    ----------
    data_dir : str
        Directory containing HDF5 episode files.
    split : str
        'train', 'validation', or 'test'.
    image_size : int
        Target image size (square).
    max_seq_len : int
        Maximum token length for language instructions.
    train_split : float
        Fraction of episodes used for training (rest for validation).
    tokenizer_name : str
        HF tokenizer name.  Set to None to skip tokenization (e.g., during tests).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 224,
        max_seq_len: int = 50,
        train_split: float = 0.8,
        tokenizer_name: Optional[str] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ) -> None:
        super().__init__()
        self.data_dir     = data_dir
        self.split        = split
        self.image_size   = image_size
        self.max_seq_len  = max_seq_len
        self.is_train     = split == "train"

        self._transform = _build_image_augmentation(image_size, self.is_train)
        self._tokenizer = None

        if tokenizer_name:
            self._tokenizer = self._load_tokenizer(tokenizer_name)

        # Load episode metadata (not all arrays — lazy loading on __getitem__)
        self._frame_index: List[Tuple[str, int]] = []  # (hdf5_path, frame_idx)
        self._episode_data: Dict[str, Dict[str, Any]] = {}  # cache
        self._load_index(data_dir, train_split)

        logger.info(
            "IndiaVLADataset[%s]: %d frames from %d episodes",
            split, len(self._frame_index), len(self._episode_data),
        )

    # ── Dataset protocol ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._frame_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hdf5_path, frame_i = self._frame_index[idx]
        ep = self._get_episode(hdf5_path)

        # Image
        frame_np = ep["frames"][frame_i]   # (H, W, C) uint8
        image    = self._transform(frame_np)  # (C, H, W) float32

        # IMU  (T, 10) → (10,)
        imu = torch.tensor(ep["imu"][frame_i], dtype=torch.float32)

        # Audio  (T, samples) → (samples,)
        audio_raw = ep["audio"][frame_i]
        audio     = torch.tensor(audio_raw, dtype=torch.float32)

        # Action  (T, 7) → (7,)
        action = torch.tensor(ep["actions"][frame_i], dtype=torch.float32)

        # Language tokens
        language_tokens, attention_mask = self._tokenize(ep["language_instruction"])

        return {
            "images":           image,
            "imu":              imu,
            "audio_features":   audio,
            "language_tokens":  language_tokens,
            "attention_mask":   attention_mask,
            "actions":          action,
        }

    # ── Index loading ────────────────────────────────────────────────────────────

    def _load_index(self, data_dir: str, train_split: float) -> None:
        """Scan HDF5 files and build (path, frame_idx) index."""
        base  = Path(data_dir)
        files = sorted(base.rglob("*.hdf5")) + sorted(base.rglob("*.h5"))
        if not files:
            logger.warning("No HDF5 files found in %s", data_dir)
            return

        # Split episodes into train/val
        n_train = max(1, int(len(files) * train_split))
        if self.is_train:
            episode_files = files[:n_train]
        else:
            episode_files = files[n_train:]

        for ep_path in episode_files:
            try:
                n_frames = self._peek_num_frames(str(ep_path))
                for fi in range(n_frames):
                    self._frame_index.append((str(ep_path), fi))
            except Exception as exc:
                logger.warning("Skipping %s: %s", ep_path, exc)

    def load_episodes(self, data_dir: str) -> List[str]:
        """Return list of episode HDF5 paths found in data_dir."""
        base  = Path(data_dir)
        files = sorted(base.rglob("*.hdf5")) + sorted(base.rglob("*.h5"))
        return [str(f) for f in files]

    # ── Episode lazy loading ────────────────────────────────────────────────────

    def _get_episode(self, hdf5_path: str) -> Dict[str, Any]:
        """Load and cache an episode's arrays."""
        if hdf5_path not in self._episode_data:
            self._episode_data[hdf5_path] = self._read_episode_arrays(hdf5_path)
        return self._episode_data[hdf5_path]

    def _read_episode_arrays(self, path: str) -> Dict[str, Any]:
        """Read all arrays from one HDF5 episode file."""
        with h5py.File(path, "r") as f:
            frames  = f["observation/images/front_camera"][:]
            audio   = f["observation/audio"][:]
            imu     = f["observation/imu"][:]
            actions = f["action"][:]
            lang_raw = f["language_instruction"][0] if "language_instruction" in f else b""
            lang = lang_raw.decode("utf-8") if isinstance(lang_raw, bytes) else str(lang_raw)
        return {
            "frames":               frames,
            "audio":                audio,
            "imu":                  imu,
            "actions":              actions,
            "language_instruction": lang,
        }

    @staticmethod
    def _peek_num_frames(path: str) -> int:
        """Read only the frame count from an HDF5 file (fast)."""
        with h5py.File(path, "r") as f:
            if "frame_index" in f:
                return len(f["frame_index"])
            if "observation/images/front_camera" in f:
                return f["observation/images/front_camera"].shape[0]
        return 0

    # ── Tokenization ────────────────────────────────────────────────────────────

    def _load_tokenizer(self, name: str):
        """Load tokenizer; return None on failure (offline / missing model)."""
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok
        except Exception as exc:
            logger.warning("Tokenizer '%s' not available: %s — using dummy encoding.", name, exc)
            return None

    def _tokenize(
        self,
        text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text → (input_ids, attention_mask) tensors."""
        if self._tokenizer is None:
            # Dummy encoding: use Unicode code-points (good enough for offline tests)
            ids  = [ord(c) % 32000 for c in text[: self.max_seq_len]]
            ids += [0] * (self.max_seq_len - len(ids))
            mask  = [1 if i < len(text) else 0 for i in range(self.max_seq_len)]
            return (
                torch.tensor(ids,  dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
            )

        enc = self._tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
        )

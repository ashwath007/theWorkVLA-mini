"""
Tests for the training module.

Covers:
  - IndiaVLADataset: load from synthetic HDF5, __len__, __getitem__ shapes
  - IndiaVLAModel:   forward pass with dummy inputs, output shape, save/load
  - Training step:   single batch backward pass doesn't crash
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

T   = 30    # frames per episode
H   = W = 32  # tiny images for fast tests
D_AUDIO = 533
D_IMU   = 10
D_ACTION = 7


def _make_episode_hdf5(path: str, episode_index: int = 0, instruction: str = "चाय बनाओ") -> None:
    """Write a minimal LeRobot HDF5 episode to path."""
    rng = np.random.default_rng(episode_index)

    frames  = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
    audio   = rng.standard_normal((T, D_AUDIO)).astype(np.float32)
    imu     = rng.standard_normal((T, D_IMU)).astype(np.float32)
    actions = rng.standard_normal((T, D_ACTION)).astype(np.float32)
    ts      = np.arange(T, dtype=np.float64) / 30.0

    with h5py.File(path, "w") as f:
        obs = f.create_group("observation")
        img = obs.create_group("images")
        img.create_dataset("front_camera", data=frames)
        obs.create_dataset("audio", data=audio)
        obs.create_dataset("imu",   data=imu)

        f.create_dataset("action",        data=actions)
        f.create_dataset("episode_index", data=np.full(T, episode_index, dtype=np.int64))
        f.create_dataset("frame_index",   data=np.arange(T, dtype=np.int64))
        f.create_dataset("timestamp",     data=ts)

        dt = h5py.special_dtype(vlen=str)
        ld = f.create_dataset("language_instruction", (1,), dtype=dt)
        ld[0] = instruction

        meta = f.create_group("metadata")
        meta.attrs["episode_index"]   = episode_index
        meta.attrs["num_frames"]      = T
        meta.attrs["lerobot_version"] = "0.5"
        meta.attrs["session_id"]      = f"test-{episode_index}"
        meta.attrs["metadata_json"]   = json.dumps({})


@pytest.fixture()
def episode_dir(tmp_path: Path) -> Path:
    """Create 5 synthetic HDF5 episode files."""
    for i in range(5):
        _make_episode_hdf5(str(tmp_path / f"episode_{i:06d}.hdf5"), episode_index=i)
    return tmp_path


@pytest.fixture()
def train_dataset(episode_dir: Path):
    from training.dataset import IndiaVLADataset
    return IndiaVLADataset(
        data_dir=str(episode_dir),
        split="train",
        image_size=H,
        max_seq_len=16,
        train_split=0.8,
        tokenizer_name=None,   # offline-safe: skip LLM tokenizer
    )


@pytest.fixture()
def small_model():
    """Create a small IndiaVLAModel that doesn't need a pretrained language model."""
    from training.model import IndiaVLAModel
    return IndiaVLAModel(
        hidden_dim=64,
        action_dim=D_ACTION,
        num_heads=2,
        dropout=0.0,
        pretrained_vision=False,    # random weights (no download)
        language_model_name="",     # triggers fallback embedding
    )


# ── IndiaVLADataset tests ─────────────────────────────────────────────────────

class TestIndiaVLADataset:

    def test_len_nonzero(self, train_dataset):
        assert len(train_dataset) > 0

    def test_len_matches_frames(self, train_dataset):
        # 4 episodes (0.8 * 5) × T frames = 4 × 30 = 120
        assert len(train_dataset) == 4 * T

    def test_getitem_returns_dict(self, train_dataset):
        item = train_dataset[0]
        assert isinstance(item, dict)

    def test_getitem_image_shape(self, train_dataset):
        item  = train_dataset[0]
        img   = item["images"]
        assert img.shape == (3, H, W), f"Expected (3,{H},{W}), got {img.shape}"
        assert img.dtype == torch.float32

    def test_getitem_imu_shape(self, train_dataset):
        item = train_dataset[0]
        imu  = item["imu"]
        assert imu.shape == (D_IMU,)

    def test_getitem_action_shape(self, train_dataset):
        item   = train_dataset[0]
        action = item["actions"]
        assert action.shape == (D_ACTION,)

    def test_getitem_language_tokens_shape(self, train_dataset):
        item   = train_dataset[0]
        tokens = item["language_tokens"]
        assert tokens.shape == (16,)  # max_seq_len
        assert tokens.dtype == torch.long

    def test_getitem_audio_is_tensor(self, train_dataset):
        item  = train_dataset[0]
        audio = item["audio_features"]
        assert isinstance(audio, torch.Tensor)
        assert audio.dtype == torch.float32

    def test_validation_split(self, episode_dir):
        from training.dataset import IndiaVLADataset
        val_ds = IndiaVLADataset(
            data_dir=str(episode_dir),
            split="validation",
            image_size=H,
            max_seq_len=16,
            train_split=0.8,
            tokenizer_name=None,
        )
        # 1 episode (5 - 4) × T = 30
        assert len(val_ds) == T


# ── IndiaVLAModel tests ───────────────────────────────────────────────────────

class TestIndiaVLAModel:

    def test_forward_output_shape(self, small_model):
        B = 2
        images = torch.zeros(B, 3, H, W)
        tokens = torch.zeros(B, 16, dtype=torch.long)
        imu    = torch.zeros(B, D_IMU)
        mask   = torch.ones(B, 16, dtype=torch.long)

        out = small_model(images, tokens, imu, mask)
        assert out.shape == (B, D_ACTION), f"Expected ({B},{D_ACTION}), got {out.shape}"

    def test_forward_output_is_float32(self, small_model):
        B  = 2
        out = small_model(
            torch.zeros(B, 3, H, W),
            torch.zeros(B, 16, dtype=torch.long),
            torch.zeros(B, D_IMU),
        )
        assert out.dtype == torch.float32

    def test_forward_no_crash_without_mask(self, small_model):
        B   = 2
        out = small_model(
            torch.randn(B, 3, H, W),
            torch.zeros(B, 16, dtype=torch.long),
            torch.randn(B, D_IMU),
        )
        assert out.shape == (B, D_ACTION)

    def test_count_parameters_returns_dict(self, small_model):
        counts = small_model.count_parameters()
        assert isinstance(counts, dict)
        for name in ("vision", "language", "imu", "fusion", "action"):
            assert name in counts
            assert isinstance(counts[name], int)

    def test_save_and_load_pretrained(self, small_model, tmp_path):
        from training.model import IndiaVLAModel

        small_model.save_pretrained(str(tmp_path))
        assert (tmp_path / "model.pt").exists()
        assert (tmp_path / "config.json").exists()

        loaded = IndiaVLAModel.from_pretrained(
            str(tmp_path),
            language_model_name="",   # fallback embedding
            pretrained_vision=False,
        )
        assert isinstance(loaded, IndiaVLAModel)

    def test_output_differs_for_different_inputs(self, small_model):
        B      = 2
        imgs1  = torch.zeros(B, 3, H, W)
        imgs2  = torch.ones(B, 3, H, W)
        tokens = torch.zeros(B, 16, dtype=torch.long)
        imu    = torch.zeros(B, D_IMU)

        out1 = small_model(imgs1, tokens, imu)
        out2 = small_model(imgs2, tokens, imu)
        assert not torch.allclose(out1, out2), "Different images should produce different actions"


# ── Training step tests ───────────────────────────────────────────────────────

class TestTrainingStep:

    def test_single_batch_backward(self, small_model):
        """A single forward + backward pass should not crash."""
        import torch.nn as nn
        from torch.optim import AdamW

        small_model.train()
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        loss_fn   = nn.MSELoss()

        B      = 4
        images = torch.rand(B, 3, H, W)
        tokens = torch.zeros(B, 16, dtype=torch.long)
        imu    = torch.rand(B, D_IMU)
        mask   = torch.ones(B, 16, dtype=torch.long)
        target = torch.rand(B, D_ACTION)

        optimizer.zero_grad()
        out  = small_model(images, tokens, imu, mask)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss), "Loss is NaN after backward"
        assert loss.item() >= 0.0

    def test_loss_decreases_over_steps(self, small_model):
        """Loss should trend down over several gradient steps on the same batch."""
        import torch.nn as nn
        from torch.optim import AdamW

        small_model.train()
        optimizer = AdamW(small_model.parameters(), lr=1e-2)
        loss_fn   = nn.MSELoss()

        B      = 4
        images = torch.rand(B, 3, H, W)
        tokens = torch.zeros(B, 16, dtype=torch.long)
        imu    = torch.rand(B, D_IMU)
        target = torch.rand(B, D_ACTION)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            out  = small_model(images, tokens, imu)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease from step 1 → step 10
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_gradient_clipping_does_not_crash(self, small_model):
        import torch.nn as nn
        from torch.optim import AdamW

        small_model.train()
        optimizer = AdamW(small_model.parameters(), lr=1e-3)
        loss_fn   = nn.MSELoss()

        B      = 2
        images = torch.rand(B, 3, H, W)
        tokens = torch.zeros(B, 16, dtype=torch.long)
        imu    = torch.rand(B, D_IMU)
        target = torch.rand(B, D_ACTION)

        optimizer.zero_grad()
        out  = small_model(images, tokens, imu)
        loss = loss_fn(out, target)
        loss.backward()
        nn.utils.clip_grad_norm_(small_model.parameters(), max_norm=1.0)
        optimizer.step()
        # No assertion needed — just ensuring no exception is raised

    def test_model_in_eval_mode_no_grad(self, small_model):
        """In eval mode, forward should not accumulate gradients."""
        small_model.eval()
        B      = 2
        images = torch.rand(B, 3, H, W)
        tokens = torch.zeros(B, 16, dtype=torch.long)
        imu    = torch.rand(B, D_IMU)

        with torch.no_grad():
            out = small_model(images, tokens, imu)

        assert out.requires_grad is False

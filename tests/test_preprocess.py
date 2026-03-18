"""
Tests for the preprocessing module.

Covers:
  - VideoPreprocessor: frame extraction (mocked), resize, normalize, face blur
  - AudioPreprocessor: load, resample, normalize, chunk
  - IMUPreprocessor:   load, gravity removal, normalization, interpolation
  - HDF5Writer:        write/read round-trip
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def fake_frames() -> list:
    """10 random BGR uint8 frames at 64×64."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]


@pytest.fixture()
def fake_audio_wav(tmp_dir: Path) -> Path:
    """Write a 1-second 16kHz mono sine WAV and return its path."""
    sr    = 16000
    freq  = 440.0
    t     = np.linspace(0, 1, sr, endpoint=False)
    pcm   = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    path  = tmp_dir / "audio.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


@pytest.fixture()
def fake_imu_csv(tmp_dir: Path) -> Path:
    """Write 300 rows of IMU data at ~100 Hz and return its path."""
    n    = 300
    t    = 1700000000.0 + np.arange(n) * 0.01
    rng  = np.random.default_rng(1)
    data = {
        "timestamp": t,
        "accel_x":   rng.normal(0.0,  0.3, n),
        "accel_y":   rng.normal(0.0,  0.3, n),
        "accel_z":   rng.normal(9.81, 0.1, n),
        "gyro_x":    rng.normal(0.0, 0.05, n),
        "gyro_y":    rng.normal(0.0, 0.05, n),
        "gyro_z":    rng.normal(0.0, 0.05, n),
    }
    df   = pd.DataFrame(data)
    path = tmp_dir / "imu.csv"
    df.to_csv(path, index=False)
    return path


# ── VideoPreprocessor tests ───────────────────────────────────────────────────

class TestVideoPreprocessor:

    def test_resize_frames(self, fake_frames):
        from preprocess.video import VideoPreprocessor
        vp = VideoPreprocessor()
        resized = vp.resize_frames(fake_frames, target_size=(32, 32))
        assert len(resized) == len(fake_frames)
        for f in resized:
            assert f.shape == (32, 32, 3), f"Expected (32,32,3), got {f.shape}"

    def test_normalize_frames_range(self, fake_frames):
        from preprocess.video import VideoPreprocessor
        vp = VideoPreprocessor()
        arr = vp.normalize_frames(fake_frames)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0 - 1e-6
        assert arr.max() <= 1.0 + 1e-6

    def test_normalize_frames_shape(self, fake_frames):
        from preprocess.video import VideoPreprocessor
        vp = VideoPreprocessor()
        arr = vp.normalize_frames(fake_frames)
        # Should be (T, H, W, 3) but BGR→RGB means channel order changes
        assert arr.shape[0] == len(fake_frames)
        assert arr.shape[-1] == 3

    def test_blur_faces_returns_same_count(self, fake_frames):
        from preprocess.video import VideoPreprocessor
        vp = VideoPreprocessor()
        blurred = vp.blur_faces(fake_frames)
        assert len(blurred) == len(fake_frames)

    def test_blur_faces_output_shape(self, fake_frames):
        from preprocess.video import VideoPreprocessor
        vp = VideoPreprocessor()
        blurred = vp.blur_faces(fake_frames)
        for orig, blur in zip(fake_frames, blurred):
            assert orig.shape == blur.shape, "Shape changed after face blur"

    def test_extract_frames_mock(self, tmp_dir, monkeypatch):
        """Test extract_frames using monkeypatched cv2.VideoCapture."""
        import cv2

        class MockCapture:
            def __init__(self, path): self._i = 0
            def isOpened(self): return True
            def get(self, prop):
                if prop == cv2.CAP_PROP_FPS: return 30.0
                if prop == cv2.CAP_PROP_FRAME_COUNT: return 5
                return 0
            def read(self):
                if self._i < 5:
                    self._i += 1
                    return True, np.zeros((32, 32, 3), dtype=np.uint8)
                return False, None
            def release(self): pass

        monkeypatch.setattr(cv2, "VideoCapture", MockCapture)

        from preprocess.video import VideoPreprocessor
        vp = VideoPreprocessor()
        frames, timestamps = vp.extract_frames("dummy.mp4")
        assert len(frames) == 5
        assert len(timestamps) == 5
        assert timestamps[1] == pytest.approx(1 / 30.0, abs=1e-4)


# ── AudioPreprocessor tests ───────────────────────────────────────────────────

class TestAudioPreprocessor:

    def test_load_audio_shape(self, fake_audio_wav):
        from preprocess.audio import AudioPreprocessor
        ap = AudioPreprocessor()
        samples, sr = ap.load_audio(str(fake_audio_wav))
        assert sr == 16000
        assert samples.dtype == np.float32
        assert len(samples) == 16000

    def test_load_audio_range(self, fake_audio_wav):
        from preprocess.audio import AudioPreprocessor
        ap = AudioPreprocessor()
        samples, _ = ap.load_audio(str(fake_audio_wav))
        assert samples.min() >= -1.0 - 1e-5
        assert samples.max() <= 1.0  + 1e-5

    def test_resample_to_16k_noop(self):
        from preprocess.audio import AudioPreprocessor
        ap = AudioPreprocessor()
        src = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
        out = ap.resample_to_16k(src, 16000)
        np.testing.assert_allclose(out, src)

    def test_resample_changes_length(self):
        from preprocess.audio import AudioPreprocessor
        ap = AudioPreprocessor()
        src = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
        out = ap.resample_to_16k(src, 44100, target_sr=16000)
        expected_len = int(len(src) * 16000 / 44100)
        assert abs(len(out) - expected_len) <= 2  # allow 2 sample tolerance

    def test_normalize_peak(self):
        from preprocess.audio import AudioPreprocessor
        ap  = AudioPreprocessor()
        src = np.array([0.0, 0.5, -0.25, 0.1], dtype=np.float32)
        out = ap.normalize_audio(src)
        assert abs(np.max(np.abs(out)) - 1.0) < 1e-5

    def test_chunk_by_timestamps(self):
        from preprocess.audio import AudioPreprocessor
        ap         = AudioPreprocessor()
        sr         = 16000
        fps        = 30
        n_frames   = 5
        samples    = np.random.default_rng(0).standard_normal(sr * 2).astype(np.float32)
        timestamps = np.arange(n_frames) / fps
        chunks     = ap.chunk_by_timestamps(samples, sr, timestamps, audio_start_time=0.0)
        assert len(chunks) == n_frames
        spf = sr // fps
        for c in chunks:
            assert len(c) == spf


# ── IMUPreprocessor tests ─────────────────────────────────────────────────────

class TestIMUPreprocessor:

    def test_load_imu_returns_dataframe(self, fake_imu_csv):
        from preprocess.imu import IMUPreprocessor
        ip = IMUPreprocessor()
        df = ip.load_imu(str(fake_imu_csv))
        assert len(df) == 300
        for col in ["timestamp", "accel_x", "accel_y", "accel_z",
                    "gyro_x", "gyro_y", "gyro_z"]:
            assert col in df.columns

    def test_normalize_imu_zero_mean(self, fake_imu_csv):
        from preprocess.imu import IMUPreprocessor
        ip = IMUPreprocessor()
        df = ip.load_imu(str(fake_imu_csv))
        dn = ip.normalize_imu(df)
        for col in ["accel_x", "accel_y", "gyro_x"]:
            assert abs(dn[col].mean()) < 1e-5, f"Column {col} not zero-mean"

    def test_interpolate_to_fps(self, fake_imu_csv):
        from preprocess.imu import IMUPreprocessor
        ip     = IMUPreprocessor()
        df     = ip.load_imu(str(fake_imu_csv))
        df_30  = ip.interpolate_to_fps(df, target_fps=30.0)
        # Duration ≈ 300 / 100 = 3 seconds → ~90 frames at 30fps
        assert 85 <= len(df_30) <= 95

    def test_remove_gravity_reduces_mean(self, fake_imu_csv):
        from preprocess.imu import IMUPreprocessor
        ip    = IMUPreprocessor()
        df    = ip.load_imu(str(fake_imu_csv))
        accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=np.float64)
        # Before: accel_z mean ≈ 9.81
        assert abs(accel[:, 2].mean() - 9.81) < 0.5
        hp = ip.remove_gravity(accel, sample_rate=100.0)
        # After: accel_z mean should be much closer to 0
        assert abs(hp[:, 2].mean()) < 1.0

    def test_to_quaternion_shape(self):
        from preprocess.imu import IMUPreprocessor
        ip   = IMUPreprocessor()
        gyro = np.zeros((50, 3), dtype=np.float64)
        q    = ip.to_quaternion(gyro, dt=1.0 / 30)
        assert q.shape == (50, 4)
        # Identity quaternion should have w ≈ 1
        assert abs(q[0, 3] - 1.0) < 1e-3


# ── HDF5Writer round-trip test ────────────────────────────────────────────────

class TestHDF5Writer:

    def test_write_read_roundtrip(self, tmp_dir):
        from preprocess.hdf5_writer import HDF5Writer

        T   = 20
        H   = W = 32
        rng = np.random.default_rng(7)

        frames    = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
        audio     = rng.standard_normal((T, 533)).astype(np.float32)
        imu       = rng.standard_normal((T, 10)).astype(np.float32)
        ts        = np.arange(T, dtype=np.float64) / 30.0

        metadata = {
            "session_id":        "test-session",
            "start_time":        1700000000.0,
            "end_time":          1700000001.0,
            "video_fps":         30,
            "audio_sample_rate": 16000,
            "imu_hz":            100,
        }

        out_path = str(tmp_dir / "test_session.hdf5")
        writer   = HDF5Writer()
        writer.write_session(
            session_id="test-session",
            frames=frames,
            audio_chunks=audio,
            imu_data=imu,
            metadata=metadata,
            output_path=out_path,
            language_instruction="चाय बनाओ",
            timestamps=ts,
        )

        assert Path(out_path).exists(), "HDF5 file not created"

        result = writer.read_session(out_path)

        np.testing.assert_array_equal(result["frames"], frames)
        np.testing.assert_allclose(result["audio_chunks"], audio, atol=1e-5)
        np.testing.assert_allclose(result["imu_data"], imu, atol=1e-5)
        assert result["language_instruction"] == "चाय बनाओ"
        assert result["metadata"]["session_id"] == "test-session"
        assert result["metadata"]["fps"] == 30

    def test_write_creates_lerobot_schema(self, tmp_dir):
        """Verify LeRobot required fields are present."""
        import h5py
        from preprocess.hdf5_writer import HDF5Writer

        T    = 5
        rng  = np.random.default_rng(3)
        out  = str(tmp_dir / "lerobot_test.hdf5")
        w    = HDF5Writer()
        w.write_session(
            session_id="lr-test",
            frames=rng.integers(0, 255, (T, 32, 32, 3), dtype=np.uint8),
            audio_chunks=np.zeros((T, 533), dtype=np.float32),
            imu_data=np.zeros((T, 10), dtype=np.float32),
            metadata={"video_fps": 30, "start_time": 0.0, "end_time": 1.0},
            output_path=out,
        )

        with h5py.File(out, "r") as f:
            for key in ["observation/images/front_camera", "observation/audio",
                        "observation/imu", "action", "episode_index",
                        "frame_index", "timestamp", "language_instruction"]:
                assert key in f, f"Missing required key: {key}"

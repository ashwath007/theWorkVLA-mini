"""
Tests for the segmentation module.

Covers:
  - ActionSegmenter: optical flow, uniform segmentation, merge short segments
  - LeRobotChunker:  chunk format validation, episode HDF5 structure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def static_frames():
    """10 mostly-static BGR uint8 frames (little motion expected)."""
    rng    = np.random.default_rng(99)
    base   = rng.integers(100, 150, (64, 64, 3), dtype=np.uint8)
    frames = [base + rng.integers(0, 5, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    return frames


@pytest.fixture()
def moving_frames():
    """20 frames with a shifting bright square to simulate motion."""
    frames = []
    for i in range(20):
        f = np.zeros((64, 64, 3), dtype=np.uint8)
        x = (i * 3) % 50
        f[10:30, x:x+10, :] = 255  # white square moving right
        frames.append(f)
    return frames


@pytest.fixture()
def dummy_hdf5(tmp_path: Path) -> Path:
    """Create a minimal LeRobot-style HDF5 for chunker testing."""
    T   = 60
    rng = np.random.default_rng(42)
    p   = tmp_path / "session.hdf5"

    frames    = rng.integers(0, 255, (T, 32, 32, 3), dtype=np.uint8)
    audio     = np.zeros((T, 533), dtype=np.float32)
    imu       = np.zeros((T, 10),  dtype=np.float32)
    actions   = np.zeros((T, 7),   dtype=np.float32)
    ts        = np.arange(T, dtype=np.float64) / 30.0

    with h5py.File(str(p), "w") as f:
        obs = f.create_group("observation")
        img = obs.create_group("images")
        img.create_dataset("front_camera", data=frames)
        obs.create_dataset("audio",  data=audio)
        obs.create_dataset("imu",    data=imu)
        f.create_dataset("action",        data=actions)
        f.create_dataset("episode_index", data=np.zeros(T, dtype=np.int64))
        f.create_dataset("frame_index",   data=np.arange(T, dtype=np.int64))
        f.create_dataset("timestamp",     data=ts)

        dt = h5py.special_dtype(vlen=str)
        ld = f.create_dataset("language_instruction", (1,), dtype=dt)
        ld[0] = "बॉक्स उठाओ"

        meta = f.create_group("metadata")
        meta.attrs["session_id"]   = "test-001"
        meta.attrs["num_frames"]   = T
        meta.attrs["lerobot_version"] = "0.5"
        meta.attrs["metadata_json"] = json.dumps({})

    return p


# ── ActionSegmenter tests ─────────────────────────────────────────────────────

class TestActionSegmenter:

    def test_uniform_segmentation_count(self):
        from segmentation.action_segmenter import ActionSegmenter
        seg    = ActionSegmenter(fps=30)
        frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 90  # 3 seconds at 30fps
        segs   = seg.segment_uniform(frames, window_sec=1.0, fps=30)
        # 90 frames / 30 = 3 segments
        assert len(segs) == 3

    def test_uniform_segmentation_covers_all_frames(self):
        from segmentation.action_segmenter import ActionSegmenter
        seg    = ActionSegmenter(fps=30)
        n      = 97   # non-divisible
        frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * n
        segs   = seg.segment_uniform(frames, window_sec=1.0, fps=30)
        total  = sum(s.end_frame - s.start_frame for s in segs)
        assert total == n, f"Coverage gap: {total} != {n}"

    def test_uniform_segment_no_overlap(self):
        from segmentation.action_segmenter import ActionSegmenter
        seg   = ActionSegmenter(fps=30)
        frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 60
        segs  = seg.segment_uniform(frames, window_sec=1.0, fps=30)
        for i in range(len(segs) - 1):
            assert segs[i].end_frame == segs[i + 1].start_frame

    def test_optical_flow_returns_at_least_one_segment(self, moving_frames):
        from segmentation.action_segmenter import ActionSegmenter
        seg  = ActionSegmenter()
        segs = seg.segment_optical_flow(moving_frames)
        assert len(segs) >= 1

    def test_optical_flow_static_frames_low_motion(self, static_frames):
        from segmentation.action_segmenter import ActionSegmenter
        seg  = ActionSegmenter(flow_threshold_high=100.0)  # very high threshold → no boundaries
        segs = seg.segment_optical_flow(static_frames)
        assert len(segs) >= 1   # at least one (full-video fallback)
        # Static video should not produce many segments
        assert len(segs) <= 3

    def test_merge_short_segments(self):
        from segmentation.action_segmenter import ActionSegmenter, ActionSegment
        seg  = ActionSegmenter()
        segs = [
            ActionSegment(0,  5, 0.8, 2.0),   # short (5 frames < min 15)
            ActionSegment(5, 40, 0.9, 3.0),   # long
            ActionSegment(40, 50, 0.7, 1.5),  # short (10 frames < min 15)
        ]
        merged = seg.merge_short_segments(segs, min_frames=15)
        # Short segments should be merged
        assert all(s.num_frames >= 15 for s in merged) or len(merged) == 1

    def test_segment_confidence_in_range(self, moving_frames):
        from segmentation.action_segmenter import ActionSegmenter
        seg  = ActionSegmenter()
        segs = seg.segment_optical_flow(moving_frames)
        for s in segs:
            assert 0.0 <= s.confidence <= 1.0, f"Confidence out of range: {s.confidence}"

    def test_segment_frames_positive(self, moving_frames):
        from segmentation.action_segmenter import ActionSegmenter
        seg  = ActionSegmenter()
        segs = seg.segment_optical_flow(moving_frames)
        for s in segs:
            assert s.num_frames > 0

    def test_action_segment_to_dict(self):
        from segmentation.action_segmenter import ActionSegment
        s = ActionSegment(start_frame=0, end_frame=30, confidence=0.75, motion_magnitude=2.5)
        d = s.to_dict()
        assert d["start_frame"] == 0
        assert d["end_frame"] == 30
        assert d["confidence"] == pytest.approx(0.75, abs=1e-3)


# ── LeRobotChunker tests ──────────────────────────────────────────────────────

class TestLeRobotChunker:

    def test_chunk_yields_episodes(self, dummy_hdf5, tmp_path):
        from segmentation.lerobot_chunker import LeRobotChunker
        from segmentation.action_segmenter import ActionSegment

        segs = [
            ActionSegment(0, 20, 0.9, 3.0),
            ActionSegment(20, 40, 0.8, 2.5),
            ActionSegment(40, 60, 0.7, 2.0),
        ]
        chunker  = LeRobotChunker()
        episodes = list(chunker.chunk_session(str(dummy_hdf5), segs, []))
        assert len(episodes) == 3

    def test_episode_has_correct_fields(self, dummy_hdf5):
        from segmentation.lerobot_chunker import LeRobotChunker
        from segmentation.action_segmenter import ActionSegment

        segs     = [ActionSegment(0, 30, 0.9, 2.0)]
        chunker  = LeRobotChunker()
        episodes = list(chunker.chunk_session(str(dummy_hdf5), segs, []))
        assert len(episodes) == 1
        ep = episodes[0]

        assert "image" in ep.observation
        assert "audio" in ep.observation
        assert "imu"   in ep.observation
        assert ep.action.shape == (30, 7)
        assert len(ep.timestamps) == 30
        assert isinstance(ep.language_instruction, str)
        assert isinstance(ep.episode_id, str)

    def test_short_segment_skipped(self, dummy_hdf5):
        from segmentation.lerobot_chunker import LeRobotChunker
        from segmentation.action_segmenter import ActionSegment

        segs    = [ActionSegment(0, 3, 0.9, 1.0)]  # 3 frames < min 10
        chunker = LeRobotChunker(min_frames_per_episode=10)
        eps     = list(chunker.chunk_session(str(dummy_hdf5), segs, []))
        assert len(eps) == 0, "Short segment should have been skipped"

    def test_save_episodes_creates_hdf5(self, dummy_hdf5, tmp_path):
        from segmentation.lerobot_chunker import LeRobotChunker
        from segmentation.action_segmenter import ActionSegment

        segs     = [ActionSegment(0, 30, 0.9, 2.0), ActionSegment(30, 60, 0.8, 1.5)]
        chunker  = LeRobotChunker()
        episodes = list(chunker.chunk_session(str(dummy_hdf5), segs, []))
        out_dir  = tmp_path / "episodes"
        paths    = chunker.save_episodes(episodes, str(out_dir))

        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()
            assert p.endswith(".hdf5")

    def test_saved_episode_hdf5_has_lerobot_keys(self, dummy_hdf5, tmp_path):
        from segmentation.lerobot_chunker import LeRobotChunker
        from segmentation.action_segmenter import ActionSegment

        segs     = [ActionSegment(0, 30, 0.9, 2.0)]
        chunker  = LeRobotChunker()
        episodes = list(chunker.chunk_session(str(dummy_hdf5), segs, []))
        out_dir  = tmp_path / "episodes"
        paths    = chunker.save_episodes(episodes, str(out_dir))

        with h5py.File(paths[0], "r") as f:
            required_keys = [
                "observation/images/front_camera",
                "observation/audio",
                "observation/imu",
                "action",
                "episode_index",
                "frame_index",
                "timestamp",
                "language_instruction",
            ]
            for k in required_keys:
                assert k in f, f"Missing key in saved episode: {k}"

    def test_create_dataset_info_json(self, dummy_hdf5, tmp_path):
        from segmentation.lerobot_chunker import LeRobotChunker
        from segmentation.action_segmenter import ActionSegment

        segs     = [ActionSegment(0, 30, 0.9, 2.0)]
        chunker  = LeRobotChunker()
        episodes = list(chunker.chunk_session(str(dummy_hdf5), segs, []))
        out_dir  = tmp_path / "episodes"
        chunker.save_episodes(episodes, str(out_dir))
        info     = chunker.create_dataset_info(episodes, str(out_dir))

        assert info["total_episodes"] == 1
        assert "features" in info
        assert "observation.images.front_camera" in info["features"]
        assert (out_dir / "dataset_info.json").exists()

    def test_episode_index_increments(self, dummy_hdf5):
        from segmentation.lerobot_chunker import LeRobotChunker
        from segmentation.action_segmenter import ActionSegment

        segs    = [ActionSegment(0, 20, 0.9, 2.0), ActionSegment(20, 40, 0.8, 2.0)]
        chunker = LeRobotChunker()
        eps     = list(chunker.chunk_session(str(dummy_hdf5), segs, [], base_episode_index=5))

        assert eps[0].episode_index == 5
        assert eps[1].episode_index == 6

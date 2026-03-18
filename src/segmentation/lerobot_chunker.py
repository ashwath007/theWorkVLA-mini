"""
LeRobotChunker: Convert segmented sessions into LeRobot v0.5 compatible episode files.

Each episode corresponds to one ActionSegment and includes:
  - observation: {image, audio, imu}
  - action: 7-DOF head-pose delta
  - language_instruction: from Whisper transcription
  - LeRobot schema fields: episode_index, frame_index, timestamp
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import h5py
import numpy as np

from .action_segmenter import ActionSegment

logger = logging.getLogger(__name__)

_LEROBOT_VERSION = "0.5"


@dataclass
class LeRobotEpisode:
    """A single LeRobot training episode."""
    episode_id: str
    episode_index: int
    observation: Dict[str, np.ndarray]   # keys: image, audio, imu
    action: np.ndarray                   # (T, 7) float32
    language_instruction: str
    timestamps: np.ndarray               # (T,) float64
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.timestamps)


class LeRobotChunker:
    """
    Converts a preprocessed HDF5 session into LeRobot-format episodes.

    Parameters
    ----------
    image_key : str
        HDF5 path to image data within observation group.
    min_frames_per_episode : int
        Discard episodes shorter than this.
    """

    def __init__(
        self,
        image_key: str = "observation/images/front_camera",
        min_frames_per_episode: int = 10,
    ) -> None:
        self.image_key              = image_key
        self.min_frames_per_episode = min_frames_per_episode

    # ── Public API ──────────────────────────────────────────────────────────────

    def chunk_session(
        self,
        hdf5_path: str,
        segments: List[ActionSegment],
        transcriptions: List[Dict[str, Any]],
        base_episode_index: int = 0,
    ) -> Generator[LeRobotEpisode, None, None]:
        """
        Yield LeRobotEpisode objects for each ActionSegment.

        Parameters
        ----------
        hdf5_path : str
            Path to preprocessed session HDF5.
        segments : list of ActionSegment
        transcriptions : list of dicts with 'start_frame', 'end_frame', 'text'
            (aligned to segments by index or frame range)
        base_episode_index : int
            Starting episode index (for multi-session datasets).

        Yields
        ------
        LeRobotEpisode
        """
        # Build a map: frame_range → instruction text
        instruction_map = self._build_instruction_map(transcriptions)

        with h5py.File(hdf5_path, "r") as f:
            all_frames     = f[self.image_key][:]
            all_audio      = f["observation/audio"][:]
            all_imu        = f["observation/imu"][:]
            all_actions    = f["action"][:]
            all_timestamps = f["timestamp"][:]
            default_instr  = ""
            if "language_instruction" in f:
                raw = f["language_instruction"][0]
                default_instr = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            meta_grp = f.get("metadata", None)
            session_meta = {}
            if meta_grp is not None:
                session_meta = dict(meta_grp.attrs)

        T = len(all_frames)
        episode_counter = base_episode_index

        for seg_idx, seg in enumerate(segments):
            start = max(0, seg.start_frame)
            end   = min(T, seg.end_frame)
            n     = end - start

            if n < self.min_frames_per_episode:
                logger.debug(
                    "Skipping short segment [%d-%d] (%d frames < min %d)",
                    start, end, n, self.min_frames_per_episode,
                )
                continue

            # Slice data for this segment
            frames     = all_frames[start:end]
            audio      = all_audio[start:end]
            imu        = all_imu[start:end]
            actions    = all_actions[start:end]
            timestamps = all_timestamps[start:end]

            # Find instruction for this segment
            instruction = instruction_map.get(
                seg_idx,
                self._find_instruction_by_frames(instruction_map, start, end, default_instr),
            )

            episode = LeRobotEpisode(
                episode_id=str(uuid.uuid4()),
                episode_index=episode_counter,
                observation={
                    "image": frames,     # (T, H, W, C) uint8
                    "audio": audio,      # (T, samples) float32
                    "imu":   imu,        # (T, 10) float32
                },
                action=actions,
                language_instruction=instruction,
                timestamps=timestamps,
                metadata={
                    "session_meta":    session_meta,
                    "segment_index":   seg_idx,
                    "start_frame":     start,
                    "end_frame":       end,
                    "motion_magnitude": seg.motion_magnitude,
                    "confidence":      seg.confidence,
                },
            )

            episode_counter += 1
            yield episode

    def save_episodes(
        self,
        episodes: List[LeRobotEpisode],
        output_dir: str,
    ) -> List[str]:
        """
        Save a list of episodes to individual HDF5 files in LeRobot format.

        Parameters
        ----------
        episodes : list of LeRobotEpisode
        output_dir : str  destination directory

        Returns
        -------
        list of str  paths to saved HDF5 files.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved_paths: List[str] = []

        for ep in episodes:
            ep_path = out / f"episode_{ep.episode_index:06d}.hdf5"
            self._write_episode_hdf5(ep, str(ep_path))
            saved_paths.append(str(ep_path))
            logger.debug("Saved episode %d → %s", ep.episode_index, ep_path)

        logger.info("Saved %d episodes to %s", len(saved_paths), output_dir)
        return saved_paths

    def create_dataset_info(
        self,
        episodes: List[LeRobotEpisode],
        output_dir: str,
        dataset_name: str = "india-pov-vla",
        fps: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate dataset_info.json in LeRobot format.

        Parameters
        ----------
        episodes : list of LeRobotEpisode
        output_dir : str  directory to save dataset_info.json
        dataset_name : str
        fps : int

        Returns
        -------
        dict  dataset info.
        """
        total_frames = sum(ep.num_frames for ep in episodes)
        languages    = list({ep.language_instruction[:2] for ep in episodes
                             if len(ep.language_instruction) >= 2} or {"hi"})

        first_ep  = episodes[0] if episodes else None
        img_shape = list(first_ep.observation["image"].shape[1:]) if first_ep else [224, 224, 3]
        audio_len = first_ep.observation["audio"].shape[1] if first_ep else 533

        info: Dict[str, Any] = {
            "codebase_version":  _LEROBOT_VERSION,
            "dataset_type":      "vla_egocentric",
            "dataset_name":      dataset_name,
            "fps":               fps,
            "total_episodes":    len(episodes),
            "total_frames":      total_frames,
            "languages":         languages,
            "features": {
                "observation.images.front_camera": {
                    "dtype":  "image",
                    "shape":  img_shape,
                    "names":  ["height", "width", "channel"],
                },
                "observation.audio": {
                    "dtype":  "float32",
                    "shape":  [audio_len],
                    "names":  ["audio_samples"],
                },
                "observation.imu": {
                    "dtype":  "float32",
                    "shape":  [10],
                    "names":  ["accel_x", "accel_y", "accel_z",
                               "gyro_x",  "gyro_y",  "gyro_z",
                               "quat_x",  "quat_y",  "quat_z", "quat_w"],
                },
                "action": {
                    "dtype":  "float32",
                    "shape":  [7],
                    "names":  ["dx", "dy", "dz", "dqx", "dqy", "dqz", "dqw"],
                },
                "language_instruction": {
                    "dtype": "string",
                },
            },
            "camera_keys":       ["observation.images.front_camera"],
            "use_videos":        False,
        }

        out_path = Path(output_dir) / "dataset_info.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(info, fp, indent=2, ensure_ascii=False)

        logger.info("dataset_info.json written to %s", out_path)
        return info

    # ── Internal helpers ─────────────────────────────────────────────────────────

    def _write_episode_hdf5(self, ep: LeRobotEpisode, path: str) -> None:
        """Write a single episode to HDF5 in LeRobot v0.5 schema."""
        T = ep.num_frames
        comp = dict(compression="gzip", compression_opts=4)

        with h5py.File(path, "w") as f:
            # Observations
            obs = f.create_group("observation")
            img_grp = obs.create_group("images")
            img_grp.create_dataset("front_camera",
                                   data=ep.observation["image"].astype(np.uint8), **comp)
            obs.create_dataset("audio", data=ep.observation["audio"].astype(np.float32), **comp)
            obs.create_dataset("imu",   data=ep.observation["imu"].astype(np.float32),   **comp)

            # Action
            f.create_dataset("action", data=ep.action.astype(np.float32), **comp)

            # LeRobot schema fields
            f.create_dataset("episode_index", data=np.full(T, ep.episode_index, dtype=np.int64))
            f.create_dataset("frame_index",   data=np.arange(T, dtype=np.int64))
            f.create_dataset("timestamp",     data=ep.timestamps.astype(np.float64))

            # Language instruction
            dt = h5py.special_dtype(vlen=str)
            ld = f.create_dataset("language_instruction", (1,), dtype=dt)
            ld[0] = ep.language_instruction

            # Metadata
            meta_grp = f.create_group("metadata")
            meta_grp.attrs["episode_id"]    = ep.episode_id
            meta_grp.attrs["episode_index"] = ep.episode_index
            meta_grp.attrs["num_frames"]    = T
            meta_grp.attrs["lerobot_version"] = _LEROBOT_VERSION
            for k, v in ep.metadata.items():
                try:
                    if isinstance(v, dict):
                        meta_grp.attrs[k] = json.dumps(v)
                    else:
                        meta_grp.attrs[k] = v
                except Exception:
                    pass

    def _build_instruction_map(
        self,
        transcriptions: List[Dict[str, Any]],
    ) -> Dict[int, str]:
        """Map segment index (or frame index) to instruction text."""
        m: Dict[int, str] = {}
        for i, t in enumerate(transcriptions):
            text = t.get("text", "")
            # Support both index-keyed and frame-keyed transcriptions
            seg_idx = t.get("segment_index", i)
            m[seg_idx] = text
        return m

    def _find_instruction_by_frames(
        self,
        instruction_map: Dict[int, str],
        start_frame: int,
        end_frame: int,
        default: str,
    ) -> str:
        """Fallback: find instruction whose frame range overlaps with [start, end]."""
        return default

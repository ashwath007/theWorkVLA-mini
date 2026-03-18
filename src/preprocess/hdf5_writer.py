"""
HDF5Writer: Write and read LeRobot-compatible HDF5 episode files.

HDF5 structure (LeRobot v0.5 schema):
    /observation/images/         (T, H, W, C) uint8
    /observation/audio/          (T, samples_per_frame) float32
    /observation/imu/            (T, 10) float32  [accel xyz, gyro xyz, quat xyzw]
    /action/                     (T, 7) float32   [head dx dy dz + quat changes]
    /language_instruction         bytes (UTF-8 string)
    /episode_index               (T,) int64
    /frame_index                 (T,) int64
    /timestamp                   (T,) float64
    /metadata/                   attrs: session_id, fps, duration, language
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)

_LEROBOT_VERSION = "0.5"


class HDF5Writer:
    """
    Write multi-modal session data to a LeRobot-compatible HDF5 file.

    Parameters
    ----------
    compression : str
        HDF5 compression filter ('gzip', 'lzf', or None).
    compression_opts : int
        Compression level (1-9) for gzip.
    """

    def __init__(
        self,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
    ) -> None:
        self.compression      = compression
        self.compression_opts = compression_opts

    # ── Write ────────────────────────────────────────────────────────────────────

    def write_session(
        self,
        session_id: str,
        frames: np.ndarray,
        audio_chunks: np.ndarray,
        imu_data: np.ndarray,
        metadata: dict,
        output_path: str,
        language_instruction: str = "",
        actions: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        episode_index: int = 0,
    ) -> str:
        """
        Write a complete session to HDF5.

        Parameters
        ----------
        session_id : str
        frames : np.ndarray  (T, H, W, C) uint8
        audio_chunks : np.ndarray  (T, audio_samples) float32
        imu_data : np.ndarray  (T, 10) float32  [ax ay az gx gy gz qx qy qz qw]
        metadata : dict  from metadata.json
        output_path : str  destination .hdf5 file path
        language_instruction : str  task instruction (may be in Hindi/English)
        actions : np.ndarray  (T, 7) float32 — if None, derived from head pose changes
        timestamps : np.ndarray  (T,) float64 — if None, reconstructed from fps
        episode_index : int  LeRobot episode index

        Returns
        -------
        str  absolute path to written file.
        """
        T = len(frames)
        if T == 0:
            raise ValueError("frames array is empty; nothing to write.")

        fps        = metadata.get("video_fps", 30)
        duration   = metadata.get("end_time", 0) - metadata.get("start_time", 0)
        language   = metadata.get("language", "hi")

        if timestamps is None:
            timestamps = np.arange(T, dtype=np.float64) / fps

        if actions is None:
            actions = self._derive_actions_from_imu(imu_data)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        kw = dict(compression=self.compression, compression_opts=self.compression_opts
                  if self.compression == "gzip" else None)
        # Remove None compression_opts
        kw = {k: v for k, v in kw.items() if v is not None}

        with h5py.File(output_path, "w") as f:
            # ── Observations ────────────────────────────────────────────────────
            obs_grp = f.create_group("observation")

            imgs_grp = obs_grp.create_group("images")
            imgs_grp.create_dataset("front_camera", data=frames.astype(np.uint8), **kw)

            obs_grp.create_dataset("audio",  data=audio_chunks.astype(np.float32), **kw)
            obs_grp.create_dataset("imu",    data=imu_data.astype(np.float32),     **kw)

            # ── Actions ─────────────────────────────────────────────────────────
            f.create_dataset("action", data=actions.astype(np.float32), **kw)

            # ── Language instruction ─────────────────────────────────────────────
            dt = h5py.special_dtype(vlen=str)
            lang_ds = f.create_dataset("language_instruction", (1,), dtype=dt)
            lang_ds[0] = language_instruction

            # ── Indexing (LeRobot v0.5 required fields) ──────────────────────────
            f.create_dataset("episode_index", data=np.full(T, episode_index, dtype=np.int64))
            f.create_dataset("frame_index",   data=np.arange(T, dtype=np.int64))
            f.create_dataset("timestamp",     data=timestamps.astype(np.float64))

            # ── Metadata attributes ──────────────────────────────────────────────
            meta_grp = f.create_group("metadata")
            meta_grp.attrs["session_id"]        = session_id
            meta_grp.attrs["fps"]               = fps
            meta_grp.attrs["duration_sec"]      = duration
            meta_grp.attrs["language"]          = language
            meta_grp.attrs["lerobot_version"]   = _LEROBOT_VERSION
            meta_grp.attrs["image_height"]      = frames.shape[1] if frames.ndim >= 3 else 0
            meta_grp.attrs["image_width"]       = frames.shape[2] if frames.ndim >= 4 else 0
            meta_grp.attrs["num_frames"]        = T
            meta_grp.attrs["episode_index"]     = episode_index

            # Store full metadata dict as JSON string
            meta_grp.attrs["metadata_json"] = json.dumps(metadata)

        logger.info("Session %s written to %s (%d frames)", session_id, output_path, T)
        return str(Path(output_path).resolve())

    # ── Read ─────────────────────────────────────────────────────────────────────

    def read_session(self, hdf5_path: str) -> Dict[str, Any]:
        """
        Read a session HDF5 file back into a dict of numpy arrays.

        Returns
        -------
        dict with keys: frames, audio_chunks, imu_data, actions,
                         language_instruction, timestamps, frame_index,
                         episode_index, metadata.
        """
        with h5py.File(hdf5_path, "r") as f:
            frames        = f["observation/images/front_camera"][:]
            audio_chunks  = f["observation/audio"][:]
            imu_data      = f["observation/imu"][:]
            actions       = f["action"][:]
            lang_raw      = f["language_instruction"][0]
            timestamps    = f["timestamp"][:]
            frame_index   = f["frame_index"][:]
            episode_index = f["episode_index"][:]

            meta_grp = f["metadata"]
            metadata_json = meta_grp.attrs.get("metadata_json", "{}")
            metadata = json.loads(metadata_json)
            metadata["session_id"]      = meta_grp.attrs.get("session_id", "")
            metadata["fps"]             = meta_grp.attrs.get("fps", 30)
            metadata["duration_sec"]    = meta_grp.attrs.get("duration_sec", 0)
            metadata["language"]        = meta_grp.attrs.get("language", "hi")
            metadata["lerobot_version"] = meta_grp.attrs.get("lerobot_version", _LEROBOT_VERSION)
            metadata["num_frames"]      = meta_grp.attrs.get("num_frames", len(frames))

        language_instruction = lang_raw.decode("utf-8") if isinstance(lang_raw, bytes) else str(lang_raw)

        return {
            "frames":               frames,
            "audio_chunks":         audio_chunks,
            "imu_data":             imu_data,
            "actions":              actions,
            "language_instruction": language_instruction,
            "timestamps":           timestamps,
            "frame_index":          frame_index,
            "episode_index":        episode_index,
            "metadata":             metadata,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────────

    def _derive_actions_from_imu(self, imu_data: np.ndarray) -> np.ndarray:
        """
        Derive approximate head-pose delta actions from IMU readings.

        Action = [dx, dy, dz (position delta from accel), dqx, dqy, dqz, dqw (quaternion delta)]

        For real robotic control, this should be replaced with proper motion capture
        or forward kinematics. Here we compute a first-order approximation.

        Parameters
        ----------
        imu_data : np.ndarray  (T, 10) [ax ay az gx gy gz qx qy qz qw]

        Returns
        -------
        np.ndarray  (T, 7)  float32
        """
        T = len(imu_data)
        actions = np.zeros((T, 7), dtype=np.float32)

        if imu_data.shape[1] >= 10:
            # Quaternion changes: action[i] = q[i] - q[i-1]
            quats = imu_data[:, 6:10]    # qx qy qz qw
            accel = imu_data[:, 0:3]     # ax ay az

            # Position delta: integrate accel twice (crude — for illustration)
            actions[:, 0:3] = accel * 0.0   # zero position delta (requires double integration)
            actions[1:,  3:7] = (quats[1:] - quats[:-1])  # quaternion delta
            actions[0, 3:7]   = np.array([0, 0, 0, 1], dtype=np.float32) - quats[0]

        elif imu_data.shape[1] >= 6:
            # No quaternion column; use gyro as proxy for rotation rate
            gyro = imu_data[:, 3:6]
            actions[:, 4:7] = gyro[:, :3]

        return actions

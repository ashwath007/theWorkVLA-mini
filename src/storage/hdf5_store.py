"""
HDF5Store: High-level interface for saving and loading LeRobot episodes.

Provides listing, stats, and episode management over a directory of HDF5 files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMeta:
    """Lightweight metadata about a stored episode."""
    path: str
    session_id: str
    episode_index: int
    num_frames: int
    duration_sec: float
    language_instruction: str
    lerobot_version: str = "0.5"


@dataclass
class DatasetStats:
    """Aggregate statistics for a data directory."""
    total_episodes: int
    total_frames: int
    total_duration_sec: float
    sessions: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)


class HDF5Store:
    """
    Manages a directory of LeRobot-format HDF5 episode files.

    Parameters
    ----------
    compression : str
        HDF5 compression filter ('gzip', 'lzf', or None).
    compression_opts : int
        Gzip compression level (1-9).
    """

    def __init__(
        self,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
    ) -> None:
        self.compression      = compression
        self.compression_opts = compression_opts

    # ── Save ─────────────────────────────────────────────────────────────────────

    def save_episode(self, episode: Dict[str, Any], path: str) -> str:
        """
        Save an episode dict to HDF5.

        The episode dict must contain:
            frames              np.ndarray (T, H, W, C) uint8
            audio_chunks        np.ndarray (T, samples) float32
            imu_data            np.ndarray (T, 10) float32
            actions             np.ndarray (T, 7) float32
            language_instruction str
            timestamps          np.ndarray (T,) float64
            episode_index       int
            metadata            dict (optional)

        Parameters
        ----------
        episode : dict
        path : str  destination .hdf5 path

        Returns
        -------
        str  resolved absolute path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        kw = {"compression": self.compression}
        if self.compression == "gzip":
            kw["compression_opts"] = self.compression_opts

        frames     = np.asarray(episode["frames"],       dtype=np.uint8)
        audio      = np.asarray(episode["audio_chunks"], dtype=np.float32)
        imu        = np.asarray(episode["imu_data"],     dtype=np.float32)
        actions    = np.asarray(episode["actions"],      dtype=np.float32)
        timestamps = np.asarray(episode.get("timestamps", np.arange(len(frames), dtype=np.float64)))
        ep_idx     = int(episode.get("episode_index", 0))
        T          = len(frames)
        meta       = episode.get("metadata", {})
        lang       = episode.get("language_instruction", "")

        with h5py.File(path, "w") as f:
            obs = f.create_group("observation")
            img_grp = obs.create_group("images")
            img_grp.create_dataset("front_camera", data=frames,  **kw)
            obs.create_dataset("audio", data=audio,  **kw)
            obs.create_dataset("imu",   data=imu,    **kw)

            f.create_dataset("action",        data=actions,                          **kw)
            f.create_dataset("episode_index", data=np.full(T, ep_idx, dtype=np.int64))
            f.create_dataset("frame_index",   data=np.arange(T, dtype=np.int64))
            f.create_dataset("timestamp",     data=timestamps.astype(np.float64))

            dt = h5py.special_dtype(vlen=str)
            ld = f.create_dataset("language_instruction", (1,), dtype=dt)
            ld[0] = lang

            meta_grp = f.create_group("metadata")
            meta_grp.attrs["episode_index"]   = ep_idx
            meta_grp.attrs["num_frames"]      = T
            meta_grp.attrs["lerobot_version"] = "0.5"
            meta_grp.attrs["metadata_json"]   = json.dumps(meta)
            if "session_id" in meta:
                meta_grp.attrs["session_id"] = meta["session_id"]

        logger.debug("Episode %d saved → %s (%d frames)", ep_idx, path, T)
        return str(Path(path).resolve())

    # ── Load ─────────────────────────────────────────────────────────────────────

    def load_episode(self, path: str) -> Dict[str, Any]:
        """
        Load an episode from HDF5 and return as a dict.

        Returns
        -------
        dict with keys: frames, audio_chunks, imu_data, actions,
                         language_instruction, timestamps, episode_index, metadata.
        """
        with h5py.File(path, "r") as f:
            frames     = f["observation/images/front_camera"][:]
            audio      = f["observation/audio"][:]
            imu        = f["observation/imu"][:]
            actions    = f["action"][:]
            timestamps = f["timestamp"][:]
            ep_idx     = f["episode_index"][0] if "episode_index" in f else 0

            lang_raw = f["language_instruction"][0] if "language_instruction" in f else b""
            lang = lang_raw.decode("utf-8") if isinstance(lang_raw, bytes) else str(lang_raw)

            meta_grp = f.get("metadata", None)
            if meta_grp is not None:
                meta_json = meta_grp.attrs.get("metadata_json", "{}")
                metadata  = json.loads(meta_json)
                metadata["session_id"]      = meta_grp.attrs.get("session_id", "")
                metadata["num_frames"]      = meta_grp.attrs.get("num_frames", len(frames))
                metadata["lerobot_version"] = meta_grp.attrs.get("lerobot_version", "0.5")
            else:
                metadata = {}

        return {
            "frames":               frames,
            "audio_chunks":         audio,
            "imu_data":             imu,
            "actions":              actions,
            "language_instruction": lang,
            "timestamps":           timestamps,
            "episode_index":        int(ep_idx),
            "metadata":             metadata,
        }

    # ── List ──────────────────────────────────────────────────────────────────────

    def list_sessions(self, data_dir: str) -> List[EpisodeMeta]:
        """
        Scan data_dir recursively for HDF5 episode files and return metadata list.

        Parameters
        ----------
        data_dir : str

        Returns
        -------
        list of EpisodeMeta sorted by episode_index.
        """
        base    = Path(data_dir)
        hdf5s   = sorted(base.rglob("*.hdf5")) + sorted(base.rglob("*.h5"))
        results = []

        for p in hdf5s:
            try:
                meta = self._read_episode_meta(str(p))
                results.append(meta)
            except Exception as exc:
                logger.warning("Skipping %s: %s", p, exc)

        results.sort(key=lambda m: m.episode_index)
        logger.info("Found %d episodes in %s", len(results), data_dir)
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────────

    def get_session_stats(self, data_dir: str) -> DatasetStats:
        """
        Compute aggregate stats for all episodes in data_dir.

        Returns
        -------
        DatasetStats
        """
        episodes = self.list_sessions(data_dir)

        total_frames   = sum(e.num_frames    for e in episodes)
        total_duration = sum(e.duration_sec  for e in episodes)
        session_ids    = list(set(e.session_id for e in episodes if e.session_id))
        languages      = list(set(
            e.language_instruction[:2]
            for e in episodes
            if len(e.language_instruction) >= 2
        ))

        return DatasetStats(
            total_episodes=len(episodes),
            total_frames=total_frames,
            total_duration_sec=total_duration,
            sessions=session_ids,
            languages=languages,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────────

    def _read_episode_meta(self, path: str) -> EpisodeMeta:
        """Read only metadata attrs from an HDF5 file (without loading arrays)."""
        with h5py.File(path, "r") as f:
            meta_grp    = f.get("metadata", None)
            ep_idx      = 0
            n_frames    = 0
            session_id  = ""
            lerobot_ver = "0.5"
            duration    = 0.0

            if meta_grp is not None:
                ep_idx      = int(meta_grp.attrs.get("episode_index", 0))
                n_frames    = int(meta_grp.attrs.get("num_frames", 0))
                session_id  = str(meta_grp.attrs.get("session_id", ""))
                lerobot_ver = str(meta_grp.attrs.get("lerobot_version", "0.5"))

            if n_frames == 0 and "frame_index" in f:
                n_frames = len(f["frame_index"])

            fps = 30
            duration = n_frames / fps

            lang_raw = b""
            if "language_instruction" in f:
                try:
                    lang_raw = f["language_instruction"][0]
                except Exception:
                    pass
            lang = lang_raw.decode("utf-8") if isinstance(lang_raw, bytes) else str(lang_raw)

        return EpisodeMeta(
            path=str(path),
            session_id=session_id,
            episode_index=ep_idx,
            num_frames=n_frames,
            duration_sec=duration,
            language_instruction=lang,
            lerobot_version=lerobot_ver,
        )

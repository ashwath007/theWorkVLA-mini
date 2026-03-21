"""
DataValidator: Quality checks on assembled sessions and LeRobot episodes.

Checks each HDF5 episode for:
  - Required datasets present
  - No NaN / Inf in arrays
  - Minimum frame count
  - Image shape correct
  - Audio and IMU aligned to frame count
  - Language instruction non-empty
  - Action values in sane range
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Required HDF5 paths in a valid LeRobot episode
REQUIRED_DATASETS = [
    "observation/images",
    "observation/audio",
    "observation/imu",
    "action",
    "language_instruction",
    "frame_index",
    "timestamp",
]

MIN_FRAMES        = 5       # episodes shorter than this are invalid
MAX_FRAME_SHAPE   = (1920, 1920, 3)
MIN_FRAME_SHAPE   = (32, 32, 3)
ACTION_ABS_MAX    = 100.0   # action values outside ±100 are suspicious


class ValidationError:
    def __init__(self, path: str, check: str, detail: str) -> None:
        self.path   = path
        self.check  = check
        self.detail = detail

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "check": self.check, "detail": self.detail}


class DataValidator:
    """
    Validates HDF5 episode files produced by the pipeline.

    Parameters
    ----------
    min_frames : int
        Episodes with fewer frames are invalid.
    check_nan : bool
        Whether to scan arrays for NaN/Inf values (slower on large files).
    """

    def __init__(self, min_frames: int = MIN_FRAMES, check_nan: bool = True) -> None:
        self.min_frames = min_frames
        self.check_nan  = check_nan

    # ── Public ────────────────────────────────────────────────────────────────

    def validate_episode(self, hdf5_path: str) -> List[ValidationError]:
        """
        Run all checks on a single episode HDF5 file.

        Returns
        -------
        list of ValidationError (empty = valid).
        """
        errors: List[ValidationError] = []
        path = str(hdf5_path)

        try:
            with h5py.File(path, "r") as f:
                errors.extend(self._check_required_datasets(f, path))
                errors.extend(self._check_frame_count(f, path))
                errors.extend(self._check_image_shape(f, path))
                errors.extend(self._check_alignment(f, path))
                errors.extend(self._check_language_instruction(f, path))
                errors.extend(self._check_action_range(f, path))
                if self.check_nan:
                    errors.extend(self._check_nan_inf(f, path))
        except OSError as exc:
            errors.append(ValidationError(path, "open_file", f"Cannot open HDF5: {exc}"))

        return errors

    def validate_episodes(self, episode_paths: List[str]) -> Dict[str, Any]:
        """
        Validate a list of episode files and return an aggregate report.

        Returns
        -------
        dict with keys:
            total_count, valid_count, invalid_count,
            errors (list of dicts), valid_paths, invalid_paths
        """
        valid_paths:   List[str] = []
        invalid_paths: List[str] = []
        all_errors:    List[Dict] = []

        for path in episode_paths:
            errs = self.validate_episode(path)
            if errs:
                invalid_paths.append(path)
                all_errors.extend(e.to_dict() for e in errs)
                logger.warning("Invalid episode %s: %d error(s)", path, len(errs))
            else:
                valid_paths.append(path)

        report = {
            "total_count":   len(episode_paths),
            "valid_count":   len(valid_paths),
            "invalid_count": len(invalid_paths),
            "errors":        all_errors,
            "valid_paths":   valid_paths,
            "invalid_paths": invalid_paths,
        }
        logger.info(
            "Validation: %d/%d episodes valid",
            len(valid_paths), len(episode_paths),
        )
        return report

    def validate_session_dir(self, session_dir: str) -> Dict[str, Any]:
        """Discover and validate all episodes in a session directory."""
        base  = Path(session_dir)
        paths = sorted(str(p) for p in (base / "episodes").glob("episode_*.h5"))
        if not paths:
            paths = sorted(str(p) for p in base.glob("**/*.h5"))
        return self.validate_episodes(paths)

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_required_datasets(
        self, f: h5py.File, path: str
    ) -> List[ValidationError]:
        errors = []
        for ds_path in REQUIRED_DATASETS:
            if ds_path not in f:
                errors.append(ValidationError(path, "required_dataset", f"Missing: {ds_path}"))
        return errors

    def _check_frame_count(self, f: h5py.File, path: str) -> List[ValidationError]:
        errors = []
        if "observation/images" not in f:
            return errors
        T = f["observation/images"].shape[0]
        if T < self.min_frames:
            errors.append(ValidationError(
                path, "min_frames",
                f"Only {T} frames (minimum {self.min_frames})"
            ))
        return errors

    def _check_image_shape(self, f: h5py.File, path: str) -> List[ValidationError]:
        errors = []
        if "observation/images" not in f:
            return errors
        shape = f["observation/images"].shape  # (T, H, W, C)
        if len(shape) != 4:
            errors.append(ValidationError(path, "image_shape", f"Expected 4D, got shape {shape}"))
            return errors
        H, W, C = shape[1], shape[2], shape[3]
        if C not in (1, 3, 4):
            errors.append(ValidationError(path, "image_channels", f"Unexpected channels: {C}"))
        if H < MIN_FRAME_SHAPE[0] or W < MIN_FRAME_SHAPE[1]:
            errors.append(ValidationError(
                path, "image_too_small", f"Frame {H}x{W} below minimum {MIN_FRAME_SHAPE[:2]}"
            ))
        if H > MAX_FRAME_SHAPE[0] or W > MAX_FRAME_SHAPE[1]:
            errors.append(ValidationError(
                path, "image_too_large", f"Frame {H}x{W} above max {MAX_FRAME_SHAPE[:2]}"
            ))
        return errors

    def _check_alignment(self, f: h5py.File, path: str) -> List[ValidationError]:
        """All per-frame datasets must have the same T dimension."""
        errors = []
        if "observation/images" not in f:
            return errors
        T = f["observation/images"].shape[0]
        for ds_path in ["observation/audio", "observation/imu", "action", "frame_index", "timestamp"]:
            if ds_path not in f:
                continue
            ds_T = f[ds_path].shape[0]
            if ds_T != T:
                errors.append(ValidationError(
                    path, "alignment",
                    f"{ds_path} has {ds_T} rows but images has {T}"
                ))
        return errors

    def _check_language_instruction(self, f: h5py.File, path: str) -> List[ValidationError]:
        errors = []
        if "language_instruction" not in f:
            return errors
        try:
            val = f["language_instruction"][()]
            if isinstance(val, bytes):
                val = val.decode("utf-8", errors="replace")
            elif isinstance(val, np.ndarray):
                val = val.item().decode("utf-8", errors="replace") if val.ndim == 0 else ""
            if not str(val).strip():
                errors.append(ValidationError(
                    path, "language_instruction", "Empty language instruction"
                ))
        except Exception as exc:
            errors.append(ValidationError(path, "language_instruction", f"Read error: {exc}"))
        return errors

    def _check_action_range(self, f: h5py.File, path: str) -> List[ValidationError]:
        errors = []
        if "action" not in f:
            return errors
        try:
            actions = f["action"][:]
            abs_max = float(np.abs(actions).max())
            if abs_max > ACTION_ABS_MAX:
                errors.append(ValidationError(
                    path, "action_range",
                    f"Action values exceed ±{ACTION_ABS_MAX}: max={abs_max:.2f}"
                ))
        except Exception as exc:
            errors.append(ValidationError(path, "action_range", f"Read error: {exc}"))
        return errors

    def _check_nan_inf(self, f: h5py.File, path: str) -> List[ValidationError]:
        errors = []
        float_datasets = ["observation/audio", "observation/imu", "action", "timestamp"]
        for ds_path in float_datasets:
            if ds_path not in f:
                continue
            try:
                arr = f[ds_path][:]
                if not np.issubdtype(arr.dtype, np.floating):
                    continue
                if np.any(np.isnan(arr)):
                    errors.append(ValidationError(path, "nan_values", f"NaN in {ds_path}"))
                if np.any(np.isinf(arr)):
                    errors.append(ValidationError(path, "inf_values", f"Inf in {ds_path}"))
            except Exception as exc:
                errors.append(ValidationError(path, "nan_check", f"Error reading {ds_path}: {exc}"))
        return errors

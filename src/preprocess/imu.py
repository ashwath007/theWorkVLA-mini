"""
IMUPreprocessor: Load, filter, integrate, normalize, and resample IMU data.

Follows standard MEMS IMU processing steps:
  1. Load CSV with [timestamp, accel_xyz, gyro_xyz]
  2. Remove gravity from accelerometer using a high-pass Butterworth filter
  3. Integrate gyroscope to quaternion via first-order integration
  4. Normalize per-axis to zero-mean, unit-variance
  5. Resample to match video FPS (100 Hz → 30 fps interpolation)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

_IMU_COLUMNS = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


class IMUPreprocessor:
    """
    Preprocessing utilities for raw 6-DOF IMU data (accelerometer + gyroscope).

    Parameters
    ----------
    gravity_hp_cutoff : float
        High-pass cutoff frequency (Hz) for gravity removal (default 0.5 Hz).
    gyro_unit : str
        'rad_s' or 'deg_s' — unit of incoming gyroscope data.
    """

    def __init__(
        self,
        gravity_hp_cutoff: float = 0.5,
        gyro_unit: str = "rad_s",
    ) -> None:
        self.gravity_hp_cutoff = gravity_hp_cutoff
        self.gyro_unit = gyro_unit

    # ── Load ────────────────────────────────────────────────────────────────────

    def load_imu(self, csv_path: str) -> pd.DataFrame:
        """
        Load IMU CSV and return a cleaned DataFrame.

        The CSV must have at minimum: timestamp, accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z columns.

        Returns
        -------
        pd.DataFrame with columns as above, sorted by timestamp.
        """
        df = pd.read_csv(csv_path)

        for col in _IMU_COLUMNS:
            if col not in df.columns:
                logger.warning("IMU CSV missing column '%s'; filling with zeros.", col)
                df[col] = 0.0

        df = df[_IMU_COLUMNS].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Convert gyro deg/s → rad/s if needed
        if self.gyro_unit == "deg_s":
            for col in ["gyro_x", "gyro_y", "gyro_z"]:
                df[col] = np.radians(df[col])

        logger.info("Loaded %d IMU samples from %s", len(df), csv_path)
        return df

    # ── Gravity removal ─────────────────────────────────────────────────────────

    def remove_gravity(self, accel_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Remove the static gravity component using a Butterworth high-pass filter.

        Parameters
        ----------
        accel_data : np.ndarray  shape (N, 3)  [ax, ay, az] in m/s²
        sample_rate : float  Hz

        Returns
        -------
        np.ndarray  shape (N, 3)  linear acceleration (gravity removed).
        """
        nyq    = sample_rate / 2.0
        cutoff = min(self.gravity_hp_cutoff / nyq, 0.99)
        b, a   = scipy_signal.butter(2, cutoff, btype="high", analog=False)

        result = np.zeros_like(accel_data, dtype=np.float32)
        for i in range(3):
            result[:, i] = scipy_signal.filtfilt(b, a, accel_data[:, i]).astype(np.float32)
        return result

    # ── Quaternion integration ───────────────────────────────────────────────────

    def to_quaternion(self, gyro_data: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate gyroscope (rad/s) to orientation quaternions via first-order integration.

        Parameters
        ----------
        gyro_data : np.ndarray  shape (N, 3)  angular velocity [gx, gy, gz] rad/s
        dt : float  time step in seconds (assumed uniform; use mean if variable)

        Returns
        -------
        np.ndarray  shape (N, 4)  unit quaternions [x, y, z, w].
        """
        N = len(gyro_data)
        quats = np.zeros((N, 4), dtype=np.float32)
        q = np.array([0.0, 0.0, 0.0, 1.0])   # identity: x, y, z, w

        for i in range(N):
            omega = gyro_data[i]                       # [gx, gy, gz]
            angle = np.linalg.norm(omega) * dt
            if angle > 1e-10:
                axis = omega / (np.linalg.norm(omega) + 1e-12)
                dq = Rotation.from_rotvec(axis * angle).as_quat()   # x, y, z, w
                q_rot = Rotation.from_quat(q)
                q_dlt = Rotation.from_quat(dq)
                q = (q_rot * q_dlt).as_quat()
            quats[i] = q

        return quats

    # ── Normalization ────────────────────────────────────────────────────────────

    def normalize_imu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zero-mean, unit-variance normalization for each IMU axis independently.

        Parameters
        ----------
        df : pd.DataFrame  with columns accel_x/y/z, gyro_x/y/z

        Returns
        -------
        pd.DataFrame  same schema, normalized values.
        """
        sensor_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        df_norm = df.copy()
        for col in sensor_cols:
            if col in df_norm.columns:
                mean = df_norm[col].mean()
                std  = df_norm[col].std()
                if std < 1e-8:
                    df_norm[col] = df_norm[col] - mean
                else:
                    df_norm[col] = (df_norm[col] - mean) / std
        return df_norm

    # ── Interpolation ─────────────────────────────────────────────────────────────

    def interpolate_to_fps(
        self,
        df: pd.DataFrame,
        target_fps: float = 30.0,
    ) -> pd.DataFrame:
        """
        Resample IMU data (e.g. 100 Hz) to align with video frames at target_fps.

        Uses linear interpolation for smooth values.  The timestamp column of the
        output is in the same absolute units as the input.

        Parameters
        ----------
        df : pd.DataFrame  with 'timestamp' column
        target_fps : float  desired output sample rate

        Returns
        -------
        pd.DataFrame  resampled to target_fps intervals.
        """
        if len(df) < 2:
            return df

        t_start = df["timestamp"].iloc[0]
        t_end   = df["timestamp"].iloc[-1]
        dt      = 1.0 / target_fps
        new_ts  = np.arange(t_start, t_end, dt)

        df_out = pd.DataFrame({"timestamp": new_ts})
        sensor_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        for col in sensor_cols:
            if col in df.columns:
                df_out[col] = np.interp(new_ts, df["timestamp"].values, df[col].values)
            else:
                df_out[col] = 0.0

        logger.info(
            "IMU resampled: %d → %d samples (%.1f fps)",
            len(df), len(df_out), target_fps,
        )
        return df_out

    # ── Full pipeline ─────────────────────────────────────────────────────────────

    def preprocess_pipeline(
        self,
        csv_path: str,
        target_fps: float = 30.0,
        normalize: bool = True,
        add_quaternion: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Full pipeline: load → remove gravity → resample → normalize → quaternion.

        Returns
        -------
        df_processed : pd.DataFrame  with 6 IMU channels (+ quaternion if requested)
        quats : np.ndarray  shape (N, 4) or None
        """
        df = self.load_imu(csv_path)

        # Estimate original sample rate
        dt_arr  = np.diff(df["timestamp"].values)
        orig_sr = 1.0 / np.median(dt_arr) if len(dt_arr) > 0 else 100.0

        # Remove gravity from accel
        accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=np.float64)
        df[["accel_x", "accel_y", "accel_z"]] = self.remove_gravity(accel, orig_sr)

        # Resample to target FPS
        df = self.interpolate_to_fps(df, target_fps=target_fps)

        # Quaternion integration
        quats = None
        if add_quaternion:
            gyro  = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float64)
            dt_fp = 1.0 / target_fps
            quats = self.to_quaternion(gyro, dt=dt_fp)

        if normalize:
            df = self.normalize_imu(df)

        return df, quats

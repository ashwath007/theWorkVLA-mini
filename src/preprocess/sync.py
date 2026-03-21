"""
StreamSynchronizer: Aligns video, audio, and IMU streams to a common timeline.

All timestamps are Unix epoch floats (seconds). Alignment target is the video
frame timestamps; audio and IMU are resampled/interpolated to match frame cadence
within ±10ms accuracy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StreamSynchronizer:
    """
    Aligns multi-modal streams captured during a headset session.

    Parameters
    ----------
    audio_window_sec : float
        Width (in seconds) of the audio chunk centred on each video frame.
    imu_extrapolate : bool
        Whether to extrapolate IMU at boundaries (vs. fill with last known value).
    """

    def __init__(
        self,
        audio_window_sec: float = 1.0 / 30,   # one video frame's worth
        imu_extrapolate: bool = False,
    ) -> None:
        self.audio_window_sec = audio_window_sec
        self.imu_extrapolate = imu_extrapolate

    # ── Public API ──────────────────────────────────────────────────────────────

    def align_streams(
        self,
        video_path: str,
        audio_path: str,
        imu_csv_path: str,
        metadata_path: str,
        video_timestamps_csv: Optional[str] = None,
        gps_csv_path: Optional[str] = None,
    ) -> Dict:
        """
        Align all streams and return a single dict ready for downstream processing.

        Parameters
        ----------
        video_path : str
            Path to the recorded video.mp4.
        audio_path : str
            Path to the recorded audio.wav.
        imu_csv_path : str
            Path to the recorded imu.csv.
        metadata_path : str
            Path to metadata.json (contains fps, sample_rate, etc.)
        video_timestamps_csv : str, optional
            Path to video_timestamps.csv; if not provided, timestamps are
            reconstructed from metadata fps.
        gps_csv_path : str, optional
            Path to the recorded gps.csv (1 Hz). Interpolated to video fps.

        Returns
        -------
        dict with keys:
            frames            : list of frame indices
            audio_chunks      : np.ndarray  (T, samples_per_frame)  float32
            imu_readings      : np.ndarray  (T, 6)  float32  [ax,ay,az,gx,gy,gz]
            gps_readings      : np.ndarray  (T, 6)  float32  [lat,lon,alt,speed,head_sin,head_cos]
            timestamps        : np.ndarray  (T,)    float64  Unix epoch
            frame_timestamps  : np.ndarray  (T,)    float64
            metadata          : dict
        """
        metadata = self._load_metadata(metadata_path)
        fps = metadata.get("video_fps", 30)
        sr  = metadata.get("audio_sample_rate", 16000)

        # 1. Build frame timestamps
        frame_ts = self._get_frame_timestamps(video_timestamps_csv, metadata, fps)
        T = len(frame_ts)
        logger.info("Synchronizing %d video frames at %.1f fps", T, fps)

        # 2. Align audio → per-frame audio chunks
        audio_chunks = self._align_audio(audio_path, frame_ts, sr)

        # 3. Align IMU → per-frame IMU readings
        imu_readings = self._align_imu(imu_csv_path, frame_ts)

        # 4. Align GPS → per-frame readings (optional)
        gps_readings = self._align_gps(gps_csv_path, frame_ts)

        return {
            "frames":           list(range(T)),
            "video_path":       video_path,
            "audio_chunks":     audio_chunks,            # (T, samples)
            "imu_readings":     imu_readings,            # (T, 6)
            "gps_readings":     gps_readings,            # (T, 6)
            "timestamps":       frame_ts,                # (T,)
            "frame_timestamps": frame_ts,
            "metadata":         metadata,
            "fps":              fps,
            "sample_rate":      sr,
            "num_frames":       T,
        }

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _align_gps(
        self,
        gps_csv_path: Optional[str],
        frame_ts: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate 1 Hz GPS to video frame timestamps.

        GPS columns used: lat, lon, alt, speed_kmh, heading_computed.
        Returns (T, 6) float32: [lat, lon, alt, speed_kmh, heading_sin, heading_cos].
        Returns zeros if GPS file not provided or unreadable.
        """
        T = len(frame_ts)
        result = np.zeros((T, 6), dtype=np.float32)

        if not gps_csv_path or not Path(gps_csv_path).exists():
            return result

        try:
            df = pd.read_csv(gps_csv_path)
        except Exception as exc:
            logger.warning("Cannot read GPS CSV %s: %s", gps_csv_path, exc)
            return result

        if "timestamp" not in df.columns:
            logger.warning("GPS CSV missing 'timestamp' column")
            return result

        gps_ts = df["timestamp"].to_numpy(dtype=np.float64)
        col_map = {
            0: "lat",
            1: "lon",
            2: "alt",
            3: "speed_kmh",
        }
        for col_idx, col_name in col_map.items():
            if col_name in df.columns:
                result[:, col_idx] = np.interp(
                    frame_ts, gps_ts, df[col_name].to_numpy(dtype=np.float64),
                    left=df[col_name].iloc[0], right=df[col_name].iloc[-1],
                ).astype(np.float32)

        # Cyclic encoding for heading
        heading_col = "heading_computed" if "heading_computed" in df.columns else "heading"
        if heading_col in df.columns:
            heading_rad = np.deg2rad(
                np.interp(frame_ts, gps_ts, df[heading_col].to_numpy(dtype=np.float64))
            )
            result[:, 4] = np.sin(heading_rad).astype(np.float32)
            result[:, 5] = np.cos(heading_rad).astype(np.float32)

        logger.debug("GPS aligned: shape %s", result.shape)
        return result

    def _load_metadata(self, metadata_path: str) -> dict:
        with open(metadata_path) as f:
            return json.load(f)

    def _get_frame_timestamps(
        self,
        csv_path: Optional[str],
        metadata: dict,
        fps: int,
    ) -> np.ndarray:
        """Load or reconstruct per-frame timestamps as Unix epoch float array."""
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            if "timestamp" in df.columns:
                ts = df["timestamp"].to_numpy(dtype=np.float64)
                logger.debug("Loaded %d frame timestamps from CSV", len(ts))
                return ts

        # Reconstruct from start_time and fps
        start_time = metadata.get("start_time", 0.0)
        n_frames   = metadata.get("video_frames", 0)
        if n_frames == 0:
            raise ValueError("metadata.video_frames is 0; cannot reconstruct timestamps.")
        dt = 1.0 / fps
        ts = start_time + np.arange(n_frames, dtype=np.float64) * dt
        logger.debug("Reconstructed %d frame timestamps from metadata", len(ts))
        return ts

    def _align_audio(
        self,
        audio_path: str,
        frame_ts: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Slice a WAV file into per-frame chunks.

        For each video frame at time t, we extract the audio window
        [t - half_win, t + half_win].

        Returns
        -------
        np.ndarray of shape (T, samples_per_chunk), dtype float32.
        """
        import wave as _wave

        half_win = self.audio_window_sec / 2.0
        samples_per_chunk = max(1, int(self.audio_window_sec * sample_rate))

        try:
            with _wave.open(audio_path, "rb") as wf:
                n_channels   = wf.getnchannels()
                sampwidth    = wf.getsampwidth()
                actual_sr    = wf.getframerate()
                total_frames = wf.getnframes()
                raw          = wf.readframes(total_frames)
        except Exception as exc:
            logger.error("Cannot read audio %s: %s", audio_path, exc)
            return np.zeros((len(frame_ts), samples_per_chunk), dtype=np.float32)

        # Decode bytes to float32 mono
        dtype = np.int16 if sampwidth == 2 else np.int32
        pcm   = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)

        # Normalize
        pcm /= np.iinfo(dtype).max

        # Build absolute timestamps for each audio sample
        start_time  = frame_ts[0] - half_win
        audio_ts    = start_time + np.arange(len(pcm), dtype=np.float64) / actual_sr

        chunks = np.zeros((len(frame_ts), samples_per_chunk), dtype=np.float32)
        for i, t in enumerate(frame_ts):
            t_start = t - half_win
            t_end   = t + half_win
            idx_start = int(max(0, (t_start - audio_ts[0]) * actual_sr))
            idx_end   = idx_start + samples_per_chunk
            available = pcm[idx_start:idx_end]
            n = min(len(available), samples_per_chunk)
            chunks[i, :n] = available[:n]

        logger.debug("Audio aligned: shape %s", chunks.shape)
        return chunks

    def _align_imu(
        self,
        imu_csv_path: str,
        frame_ts: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate IMU readings to video frame timestamps.

        Returns
        -------
        np.ndarray of shape (T, 6), dtype float32.
        Columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z.
        """
        imu_cols = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]

        try:
            df = pd.read_csv(imu_csv_path)
        except Exception as exc:
            logger.error("Cannot read IMU CSV %s: %s", imu_csv_path, exc)
            return np.zeros((len(frame_ts), 6), dtype=np.float32)

        if "timestamp" not in df.columns:
            raise ValueError("imu.csv must contain a 'timestamp' column.")

        imu_ts = df["timestamp"].to_numpy(dtype=np.float64)

        # Ensure we have all required columns
        for col in imu_cols:
            if col not in df.columns:
                df[col] = 0.0

        result = np.zeros((len(frame_ts), 6), dtype=np.float32)
        for j, col in enumerate(imu_cols):
            values = df[col].to_numpy(dtype=np.float64)
            interp_vals = np.interp(
                frame_ts,
                imu_ts,
                values,
                left=values[0]  if len(values) else 0.0,
                right=values[-1] if len(values) else 0.0,
            )
            result[:, j] = interp_vals.astype(np.float32)

        # Alignment accuracy check: warn if IMU coverage is sparse
        if len(imu_ts) > 0:
            max_gap = np.max(np.diff(imu_ts)) if len(imu_ts) > 1 else 0
            if max_gap > 0.01:
                logger.warning("IMU has gaps up to %.1f ms — interpolation may degrade.", max_gap * 1000)

        logger.debug("IMU aligned: shape %s", result.shape)
        return result

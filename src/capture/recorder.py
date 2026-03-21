"""
HeadsetRecorder: Captures synchronized video, audio, and IMU data from an egocentric headset.

Supports real MPU9250 IMU via smbus2 on Raspberry Pi, and simulated IMU on other platforms.
All streams are timestamped with Unix epoch (float seconds) for precise alignment.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import struct
import threading
import time
import uuid
import wave
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── GPS backend detection ──────────────────────────────────────────────────────
try:
    from .streamer import GPSReader, GPSFix  # type: ignore
    _HAS_GPS_READER = True
except ImportError:
    _HAS_GPS_READER = False
    GPSReader = None  # type: ignore[assignment,misc]

# ── IMU backend detection ──────────────────────────────────────────────────────
_HAS_SMBUS = False
try:
    import smbus2  # type: ignore
    _HAS_SMBUS = True
except ImportError:
    pass

# MPU-9250 register map (subset)
_MPU9250_ADDR = 0x68
_PWR_MGMT_1   = 0x6B
_ACCEL_XOUT_H = 0x3B
_GYRO_XOUT_H  = 0x43
_ACCEL_SCALE  = 16384.0   # ±2g → LSB/g
_GYRO_SCALE   = 131.0     # ±250°/s → LSB/(°/s)


@dataclass
class IMUSample:
    """Single IMU sample with raw sensor values."""
    timestamp: float        # Unix epoch seconds
    accel_x: float          # m/s²  (raw / scale * 9.81)
    accel_y: float
    accel_z: float
    gyro_x: float           # rad/s
    gyro_y: float
    gyro_z: float


@dataclass
class SessionMetadata:
    """Metadata saved alongside a recording session."""
    session_id: str
    start_time: float       # Unix epoch
    end_time: Optional[float]
    video_fps: int
    audio_sample_rate: int
    imu_hz: int
    video_frames: int
    audio_samples: int
    imu_samples: int
    data_dir: str


class HeadsetRecorder:
    """
    Records video, audio, and IMU streams simultaneously.

    Usage
    -----
    recorder = HeadsetRecorder(data_root="/data/sessions")
    recorder.start(session_id="my-session")
    time.sleep(60)
    recorder.stop()
    """

    def __init__(
        self,
        data_root: str = "/data/sessions",
        video_device: int = 0,
        video_fps: int = 30,
        video_width: int = 640,
        video_height: int = 480,
        audio_sample_rate: int = 16000,
        audio_channels: int = 1,
        audio_chunk_size: int = 1024,
        imu_hz: int = 100,
        autosave_interval_sec: int = 300,
        simulate_imu: Optional[bool] = None,
        gps_enabled: bool = True,
        gps_port: str = "/dev/ttyAMA0",
        gps_baud: int = 9600,
        simulate_gps: Optional[bool] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.video_device = video_device
        self.video_fps = video_fps
        self.video_width = video_width
        self.video_height = video_height
        self.audio_sample_rate = audio_sample_rate
        self.audio_channels = audio_channels
        self.audio_chunk_size = audio_chunk_size
        self.imu_hz = imu_hz
        self.autosave_interval_sec = autosave_interval_sec

        # If not specified, simulate on non-RPi platforms
        self.simulate_imu = simulate_imu if simulate_imu is not None else (not _HAS_SMBUS)

        # GPS settings
        self.gps_enabled = gps_enabled and _HAS_GPS_READER
        self.gps_port    = gps_port
        self.gps_baud    = gps_baud
        # Simulate GPS if no serial device is found and simulate not forced off
        self.simulate_gps = simulate_gps if simulate_gps is not None else (not _HAS_GPS_READER)

        # Internal state
        self._recording = False
        self._session_id: Optional[str] = None
        self._session_path: Optional[Path] = None
        self._start_time: Optional[float] = None

        # Threads
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None
        self._imu_thread: Optional[threading.Thread] = None
        self._autosave_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Buffers (filled by threads, flushed on stop / autosave)
        self._imu_samples: List[IMUSample] = []
        self._imu_lock = threading.Lock()

        # GPS reader instance (created at start() time)
        self._gps_reader: Optional[object] = None  # GPSReader when available

        # Frame / audio counters (for stats)
        self._video_frame_count = 0
        self._audio_sample_count = 0

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self, session_id: Optional[str] = None) -> str:
        """Begin recording all streams. Returns the session_id."""
        if self._recording:
            raise RuntimeError("Already recording. Call stop() first.")

        self._session_id = session_id or str(uuid.uuid4())
        self._session_path = self._get_session_path(self._session_id)
        self._session_path.mkdir(parents=True, exist_ok=True)

        self._stop_event.clear()
        self._imu_samples = []
        self._video_frame_count = 0
        self._audio_sample_count = 0
        self._start_time = time.time()
        self._recording = True

        logger.info("Starting session %s → %s", self._session_id, self._session_path)

        self._video_thread = threading.Thread(target=self._record_video, daemon=True)
        self._audio_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._imu_thread   = threading.Thread(target=self._record_imu,   daemon=True)
        self._autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)

        self._video_thread.start()
        self._audio_thread.start()
        self._imu_thread.start()
        self._autosave_thread.start()

        # GPS reader — start alongside other streams
        if self.gps_enabled and GPSReader is not None:
            self._gps_reader = GPSReader(
                port=self.gps_port,
                baud=self.gps_baud,
                simulate=self.simulate_gps,
            )
            self._gps_reader.start()
            logger.info("GPSReader started (simulate=%s)", self.simulate_gps)
        else:
            if self.gps_enabled and GPSReader is None:
                logger.warning("GPS enabled but GPSReader not available (streamer.py missing?)")
            self._gps_reader = None

        return self._session_id

    def stop(self) -> SessionMetadata:
        """Stop all recording threads and flush data to disk."""
        if not self._recording:
            raise RuntimeError("Not currently recording.")

        logger.info("Stopping session %s …", self._session_id)
        self._stop_event.set()
        self._recording = False

        # Wait for threads
        for t in (self._video_thread, self._audio_thread, self._imu_thread, self._autosave_thread):
            if t and t.is_alive():
                t.join(timeout=10)

        # Stop GPS reader
        if self._gps_reader is not None:
            try:
                self._gps_reader.stop()
            except Exception as exc:
                logger.warning("Error stopping GPSReader: %s", exc)

        # Flush remaining IMU buffer
        self._flush_imu()

        end_time = time.time()
        meta = self._write_metadata(end_time)
        logger.info("Session saved: %d frames, %d IMU samples", self._video_frame_count, len(self._imu_samples))
        return meta

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def stats(self) -> dict:
        """Live stats for display."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        gps_fix = None
        if self._gps_reader is not None:
            try:
                fix = self._gps_reader.get_latest_fix()
                if fix is not None:
                    gps_fix = {"lat": fix.lat, "lon": fix.lon, "fix_quality": fix.fix_quality}
            except Exception:
                pass
        return {
            "session_id":    self._session_id,
            "elapsed_sec":   round(elapsed, 1),
            "video_frames":  self._video_frame_count,
            "audio_samples": self._audio_sample_count,
            "imu_samples":   len(self._imu_samples),
            "gps_fix":       gps_fix,
        }

    @property
    def gps_reader(self) -> Optional[object]:
        """Return the active GPSReader instance, or None if GPS is disabled."""
        return self._gps_reader

    # ── Path helpers ────────────────────────────────────────────────────────────

    def _get_session_path(self, session_id: str) -> Path:
        """Returns /data/sessions/YYYY-MM-DD-HHMM/<session_id>/"""
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        return self.data_root / ts / session_id

    # ── Video thread ────────────────────────────────────────────────────────────

    def _record_video(self) -> None:
        """Capture frames from webcam and write to video.mp4."""
        cap = cv2.VideoCapture(self.video_device)
        if not cap.isOpened():
            logger.error("Cannot open video device %d", self.video_device)
            return

        cap.set(cv2.CAP_PROP_FPS, self.video_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)

        video_path = str(self._session_path / "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, self.video_fps, (self.video_width, self.video_height))

        timestamps_path = self._session_path / "video_timestamps.csv"
        ts_file = open(timestamps_path, "w", newline="")
        ts_writer = csv.writer(ts_file)
        ts_writer.writerow(["frame_index", "timestamp"])

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("VideoCapture read failed, retrying …")
                    time.sleep(0.01)
                    continue

                timestamp = time.time()
                out.write(frame)
                ts_writer.writerow([self._video_frame_count, timestamp])
                self._video_frame_count += 1
        finally:
            cap.release()
            out.release()
            ts_file.close()
            logger.debug("Video thread finished: %d frames", self._video_frame_count)

    # ── Audio thread ────────────────────────────────────────────────────────────

    def _record_audio(self) -> None:
        """Capture audio from microphone and write to audio.wav."""
        try:
            import pyaudio  # lazy import so non-audio systems still work
        except ImportError:
            logger.warning("pyaudio not available, skipping audio recording.")
            return

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.audio_channels,
            rate=self.audio_sample_rate,
            input=True,
            frames_per_buffer=self.audio_chunk_size,
        )

        audio_path = str(self._session_path / "audio.wav")
        wf = wave.open(audio_path, "wb")
        wf.setnchannels(self.audio_channels)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.audio_sample_rate)

        try:
            while not self._stop_event.is_set():
                data = stream.read(self.audio_chunk_size, exception_on_overflow=False)
                wf.writeframes(data)
                self._audio_sample_count += self.audio_chunk_size
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            wf.close()
            logger.debug("Audio thread finished: %d samples", self._audio_sample_count)

    # ── IMU thread ──────────────────────────────────────────────────────────────

    def _record_imu(self) -> None:
        """Capture IMU data and buffer into _imu_samples."""
        if self.simulate_imu:
            self._simulate_imu_loop()
        else:
            self._real_imu_loop()

    def _simulate_imu_loop(self) -> None:
        """Generate synthetic IMU data (gravity on Z axis + small noise)."""
        interval = 1.0 / self.imu_hz
        rng = np.random.default_rng()
        t = 0.0
        while not self._stop_event.is_set():
            ts = time.time()
            noise = rng.normal(0, 0.01, 6)
            sample = IMUSample(
                timestamp=ts,
                accel_x=noise[0],
                accel_y=noise[1],
                accel_z=9.81 + noise[2],   # gravity
                gyro_x=noise[3],
                gyro_y=noise[4],
                gyro_z=noise[5],
            )
            with self._imu_lock:
                self._imu_samples.append(sample)
            t += interval
            time.sleep(max(0, interval))

    def _real_imu_loop(self) -> None:
        """Read from MPU-9250 over I2C (smbus2)."""
        if not _HAS_SMBUS:
            logger.error("smbus2 not available; falling back to simulation.")
            self._simulate_imu_loop()
            return

        bus = smbus2.SMBus(1)
        # Wake up MPU-9250
        bus.write_byte_data(_MPU9250_ADDR, _PWR_MGMT_1, 0)
        interval = 1.0 / self.imu_hz

        while not self._stop_event.is_set():
            ts = time.time()
            raw = bus.read_i2c_block_data(_MPU9250_ADDR, _ACCEL_XOUT_H, 14)

            def to_int16(hi: int, lo: int) -> int:
                val = (hi << 8) | lo
                return val - 65536 if val > 32767 else val

            ax = to_int16(raw[0],  raw[1])  / _ACCEL_SCALE * 9.81
            ay = to_int16(raw[2],  raw[3])  / _ACCEL_SCALE * 9.81
            az = to_int16(raw[4],  raw[5])  / _ACCEL_SCALE * 9.81
            gx = to_int16(raw[8],  raw[9])  / _GYRO_SCALE  * (np.pi / 180)
            gy = to_int16(raw[10], raw[11]) / _GYRO_SCALE  * (np.pi / 180)
            gz = to_int16(raw[12], raw[13]) / _GYRO_SCALE  * (np.pi / 180)

            sample = IMUSample(timestamp=ts, accel_x=ax, accel_y=ay, accel_z=az,
                               gyro_x=gx, gyro_y=gy, gyro_z=gz)
            with self._imu_lock:
                self._imu_samples.append(sample)
            time.sleep(max(0, interval))

    # ── Autosave loop ───────────────────────────────────────────────────────────

    def _autosave_loop(self) -> None:
        """Periodically flush IMU buffer to disk to avoid data loss."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.autosave_interval_sec)
            if self._recording:
                self._flush_imu()
                logger.debug("Autosaved IMU: %d total samples", len(self._imu_samples))

    # ── Data flush ──────────────────────────────────────────────────────────────

    def _flush_imu(self) -> None:
        """Write buffered IMU samples to imu.csv (append mode)."""
        with self._imu_lock:
            samples = list(self._imu_samples)

        if not samples:
            return

        imu_path = self._session_path / "imu.csv"
        write_header = not imu_path.exists()

        with open(imu_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "accel_x", "accel_y", "accel_z",
                                  "gyro_x", "gyro_y", "gyro_z"])
            for s in samples:
                writer.writerow([s.timestamp, s.accel_x, s.accel_y, s.accel_z,
                                  s.gyro_x, s.gyro_y, s.gyro_z])

    def _write_metadata(self, end_time: float) -> SessionMetadata:
        """Persist session metadata to metadata.json."""
        with self._imu_lock:
            imu_count = len(self._imu_samples)

        meta = SessionMetadata(
            session_id=self._session_id,
            start_time=self._start_time,
            end_time=end_time,
            video_fps=self.video_fps,
            audio_sample_rate=self.audio_sample_rate,
            imu_hz=self.imu_hz,
            video_frames=self._video_frame_count,
            audio_samples=self._audio_sample_count,
            imu_samples=imu_count,
            data_dir=str(self._session_path),
        )
        meta_path = self._session_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2)

        return meta

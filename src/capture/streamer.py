"""
ChunkStreamer: Uploads completed 30-second recording chunks to the server.

Runs as a background thread on the Raspberry Pi alongside HeadsetRecorder.
Watches the session output directory, queues new chunks in SQLite, and uploads
them to the server with exponential-backoff retries. GPS data is captured by
the companion GPSReader class using the NEO-M8N module via UART.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Optional serial / NMEA backend ────────────────────────────────────────────
_HAS_SERIAL = False
try:
    import serial  # type: ignore
    _HAS_SERIAL = True
except ImportError:
    pass

_HAS_PYNMEA2 = False
try:
    import pynmea2  # type: ignore
    _HAS_PYNMEA2 = True
except ImportError:
    pass

_HAS_REQUESTS = False
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    pass


# ── GPS dataclass ─────────────────────────────────────────────────────────────

@dataclass
class GPSFix:
    """Single GPS position fix."""
    timestamp: float          # Unix epoch seconds
    lat: float                # decimal degrees, positive = North
    lon: float                # decimal degrees, positive = East
    alt: float                # metres above sea level
    speed_kmh: float          # speed over ground in km/h
    heading: float            # true heading, degrees 0-360
    fix_quality: int          # 0=no fix, 1=GPS, 2=DGPS
    num_satellites: int


# ── Chunk status constants ─────────────────────────────────────────────────────

_STATUS_PENDING    = "pending"
_STATUS_UPLOADING  = "uploading"
_STATUS_DONE       = "done"
_STATUS_FAILED     = "failed"

# Retry schedule: delays in seconds before each retry attempt
_RETRY_DELAYS: Tuple[float, ...] = (2.0, 4.0, 8.0)
_MAX_RETRIES = len(_RETRY_DELAYS)


# ── GPSReader ─────────────────────────────────────────────────────────────────

class GPSReader:
    """
    Reads NMEA sentences from a serial GPS receiver (NEO-M8N on /dev/ttyAMA0).

    Falls back to a random-walk simulation when the serial device is unavailable
    or when ``simulate=True`` is explicitly requested.

    Parameters
    ----------
    port : str
        Serial device path for the GPS module.
    baud : int
        Baud rate for the serial connection.
    simulate : bool
        Force simulation mode even if a serial device exists.
    base_lat : float
        Base latitude for the random-walk simulator (default: New Delhi).
    base_lon : float
        Base longitude for the random-walk simulator.
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baud: int = 9600,
        simulate: bool = False,
        base_lat: float = 28.6139,   # New Delhi
        base_lon: float = 77.2090,
    ) -> None:
        self.port = port
        self.baud = baud
        self.simulate = simulate or not _HAS_SERIAL or not _HAS_PYNMEA2
        self._base_lat = base_lat
        self._base_lon = base_lon

        self._lock = threading.Lock()
        self._latest_fix: Optional[GPSFix] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Simulation state
        self._sim_lat = base_lat
        self._sim_lon = base_lon
        self._sim_heading = random.uniform(0, 360)
        self._sim_speed = random.uniform(0, 30)  # km/h

        if self.simulate:
            logger.info("GPSReader: simulation mode (port=%s unavailable or simulate=True)", port)

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the GPS reading thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        target = self._simulate_loop if self.simulate else self._serial_loop
        self._thread = threading.Thread(target=target, daemon=True, name="gps-reader")
        self._thread.start()
        logger.debug("GPSReader started (simulate=%s)", self.simulate)

    def stop(self) -> None:
        """Stop the GPS reading thread and wait for it to finish."""
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.debug("GPSReader stopped")

    def get_latest_fix(self) -> Optional[GPSFix]:
        """Return the most recently obtained GPS fix (thread-safe)."""
        with self._lock:
            return self._latest_fix

    def write_chunk_csv(
        self,
        output_path: str,
        chunk_start: float,
        chunk_end: float,
        fixes: List[GPSFix],
    ) -> None:
        """
        Write a list of GPSFix objects to a CSV file for one chunk.

        Parameters
        ----------
        output_path : str
            Destination file path.
        chunk_start : float
            Chunk start time as Unix epoch; used to filter fixes.
        chunk_end : float
            Chunk end time as Unix epoch.
        fixes : list of GPSFix
            All fixes accumulated during the chunk window.
        """
        fieldnames = [
            "timestamp", "lat", "lon", "alt",
            "speed_kmh", "heading", "fix_quality", "num_satellites",
        ]
        chunk_fixes = [f for f in fixes if chunk_start <= f.timestamp <= chunk_end]

        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for fix in chunk_fixes:
                writer.writerow(asdict(fix))

        logger.debug("GPS CSV written: %d fixes → %s", len(chunk_fixes), output_path)

    # ── Serial reading loop ─────────────────────────────────────────────────────

    def _serial_loop(self) -> None:
        """Read NMEA sentences from the serial port and parse them."""
        try:
            ser = serial.Serial(self.port, self.baud, timeout=2.0)
        except Exception as exc:
            logger.warning("GPSReader: cannot open %s (%s), falling back to simulation", self.port, exc)
            self.simulate = True
            self._simulate_loop()
            return

        logger.info("GPSReader: connected to %s @ %d baud", self.port, self.baud)
        rmc_data: Dict = {}
        gga_data: Dict = {}

        try:
            while not self._stop_event.is_set():
                try:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                except Exception as exc:
                    logger.warning("GPSReader serial read error: %s", exc)
                    time.sleep(0.1)
                    continue

                if not line.startswith("$"):
                    continue

                try:
                    msg = pynmea2.parse(line)
                except Exception:
                    continue

                ts = time.time()

                if isinstance(msg, pynmea2.types.talker.RMC):
                    if msg.status == "A":  # Active / valid fix
                        rmc_data = {
                            "lat": msg.latitude,
                            "lon": msg.longitude,
                            "speed_kmh": float(msg.spd_over_grnd or 0) * 1.852,  # knots → km/h
                            "heading": float(msg.true_course or 0),
                        }

                elif isinstance(msg, pynmea2.types.talker.GGA):
                    gga_data = {
                        "alt": float(msg.altitude or 0),
                        "fix_quality": int(msg.gps_qual or 0),
                        "num_satellites": int(msg.num_sats or 0),
                    }

                # Merge RMC + GGA when both are available
                if rmc_data and gga_data:
                    fix = GPSFix(
                        timestamp=ts,
                        lat=rmc_data["lat"],
                        lon=rmc_data["lon"],
                        alt=gga_data["alt"],
                        speed_kmh=rmc_data["speed_kmh"],
                        heading=rmc_data["heading"],
                        fix_quality=gga_data["fix_quality"],
                        num_satellites=gga_data["num_satellites"],
                    )
                    with self._lock:
                        self._latest_fix = fix
                    # Clear so next cycle waits for fresh pair
                    rmc_data = {}
                    gga_data = {}

        finally:
            try:
                ser.close()
            except Exception:
                pass

    # ── Simulation loop ─────────────────────────────────────────────────────────

    def _simulate_loop(self) -> None:
        """Generate a random-walk GPS trace at 1 Hz."""
        rng = random.Random()
        while not self._stop_event.is_set():
            # Random walk: small heading perturbation, then advance position
            self._sim_heading = (self._sim_heading + rng.gauss(0, 5)) % 360
            self._sim_speed = max(0, self._sim_speed + rng.gauss(0, 2))

            # Convert heading to radians
            heading_rad = math.radians(self._sim_heading)
            # Approx degrees per metre at equator
            metres_per_deg_lat = 111320.0
            metres_per_deg_lon = 111320.0 * math.cos(math.radians(self._sim_lat))

            # Distance advanced per second at current speed
            dist_m = self._sim_speed / 3.6  # km/h → m/s

            self._sim_lat += (dist_m * math.cos(heading_rad)) / metres_per_deg_lat
            self._sim_lon += (dist_m * math.sin(heading_rad)) / metres_per_deg_lon

            fix = GPSFix(
                timestamp=time.time(),
                lat=self._sim_lat,
                lon=self._sim_lon,
                alt=rng.uniform(200, 250),
                speed_kmh=self._sim_speed,
                heading=self._sim_heading,
                fix_quality=1,
                num_satellites=rng.randint(6, 12),
            )
            with self._lock:
                self._latest_fix = fix

            self._stop_event.wait(timeout=1.0)


# ── ChunkStreamer ──────────────────────────────────────────────────────────────

class ChunkStreamer:
    """
    Watches the session output directory for completed chunks and uploads them.

    Each chunk consists of five files:
    ``video_NNN.mp4``, ``audio_NNN.wav``, ``imu_NNN.csv``,
    ``gps_NNN.csv``, ``chunk_meta_NNN.json``.

    Unsent chunks are persisted in a local SQLite queue so no data is lost if
    WiFi is unavailable.  On reconnect the queue is drained in chunk-index order.

    Parameters
    ----------
    server_url : str
        Base URL of the server, e.g. ``http://192.168.1.100:8000``.
    session_id : str
        Session identifier sent with every upload request.
    data_dir : str
        Directory to watch for completed chunk files.
    api_key : str, optional
        Bearer token added to every request as ``Authorization: Bearer <key>``.
    db_path : str
        Path to the SQLite queue database.
    poll_interval : float
        How often (seconds) to scan for new chunk files.
    """

    _CHUNK_FILES = ("video_{n:04d}.mp4", "audio_{n:04d}.wav",
                    "imu_{n:04d}.csv", "gps_{n:04d}.csv", "chunk_meta_{n:04d}.json")

    def __init__(
        self,
        server_url: str,
        session_id: str,
        data_dir: str,
        api_key: Optional[str] = None,
        db_path: str = "/tmp/vla_upload_queue.db",
        poll_interval: float = 5.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self.data_dir = Path(data_dir)
        self.api_key = api_key
        self.db_path = db_path
        self.poll_interval = poll_interval

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._db_lock = threading.Lock()

        # Stats counters
        self._uploaded = 0
        self._failed = 0

        self._init_db()

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background streamer thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("ChunkStreamer already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="chunk-streamer"
        )
        self._thread.start()
        logger.info("ChunkStreamer started: watching %s → %s", self.data_dir, self.server_url)

    def stop(self) -> None:
        """Signal the streamer thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        logger.info("ChunkStreamer stopped")

    def get_stats(self) -> Dict[str, int]:
        """Return current upload statistics.

        Returns
        -------
        dict with keys ``queued``, ``uploaded``, ``failed``.
        """
        queued = self._count_pending()
        return {
            "queued":   queued,
            "uploaded": self._uploaded,
            "failed":   self._failed,
        }

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Continuously discover new chunks and drain the upload queue."""
        while not self._stop_event.is_set():
            try:
                self._discover_chunks()
                self._drain_queue()
            except Exception as exc:
                logger.error("ChunkStreamer loop error: %s", exc, exc_info=True)
            self._stop_event.wait(timeout=self.poll_interval)

    def _discover_chunks(self) -> None:
        """Scan data_dir for complete chunk sets and add new ones to the queue."""
        if not self.data_dir.exists():
            return

        # Find the highest chunk index already in DB
        known_indices = self._get_known_indices()

        # Walk candidate chunk indices
        chunk_index = 0
        while True:
            files = self._chunk_file_paths(chunk_index)
            if not all(p.exists() for p in files.values()):
                # If no file at this index we may just not be there yet
                if chunk_index > max(known_indices, default=-1) + 5:
                    break  # Stop scanning well beyond known range
                chunk_index += 1
                continue

            if chunk_index not in known_indices:
                self._enqueue_chunk(chunk_index, files)
                logger.info("Queued chunk %d for upload", chunk_index)

            chunk_index += 1

    def _drain_queue(self) -> None:
        """Upload all pending chunks from the SQLite queue."""
        pending = self._fetch_pending_chunks()
        for row in pending:
            chunk_index = row["chunk_index"]
            if self._stop_event.is_set():
                break
            self._upload_chunk_with_retry(chunk_index)

    # ── Upload logic ───────────────────────────────────────────────────────────

    def _upload_chunk_with_retry(self, chunk_index: int) -> bool:
        """
        Attempt to upload a chunk with exponential-backoff retries.

        Returns True if the upload succeeded.
        """
        if not _HAS_REQUESTS:
            logger.error("requests library not available; cannot upload")
            return False

        files = self._chunk_file_paths(chunk_index)
        if not all(p.exists() for p in files.values()):
            logger.warning("Chunk %d files missing; marking failed", chunk_index)
            self._update_status(chunk_index, _STATUS_FAILED)
            self._failed += 1
            return False

        self._update_status(chunk_index, _STATUS_UPLOADING)

        for attempt, delay in enumerate(_RETRY_DELAYS):
            try:
                success = self._do_upload(chunk_index, files)
                if success:
                    self._update_status(chunk_index, _STATUS_DONE)
                    self._uploaded += 1
                    logger.info("Chunk %d uploaded successfully", chunk_index)
                    return True
                else:
                    logger.warning(
                        "Chunk %d upload attempt %d/%d failed (non-200); retrying in %.0fs",
                        chunk_index, attempt + 1, _MAX_RETRIES, delay,
                    )
            except Exception as exc:
                logger.warning(
                    "Chunk %d upload attempt %d/%d error: %s; retrying in %.0fs",
                    chunk_index, attempt + 1, _MAX_RETRIES, exc, delay,
                )

            # Wait before retry, but respect stop_event
            if self._stop_event.wait(timeout=delay):
                break  # Stopping, abort retries

        self._update_status(chunk_index, _STATUS_FAILED)
        self._failed += 1
        logger.error("Chunk %d failed after %d retries", chunk_index, _MAX_RETRIES)
        return False

    def _do_upload(self, chunk_index: int, files: Dict[str, Path]) -> bool:
        """Perform the actual HTTP multipart POST for one chunk."""
        url = f"{self.server_url}/api/ingest/chunk"
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Read meta file for device_id
        device_id = "rpi5-headset"
        try:
            meta_path = files["chunk_meta"]
            with open(meta_path) as fh:
                meta = json.load(fh)
            device_id = meta.get("device_id", device_id)
        except Exception:
            pass

        open_files = []
        try:
            multipart_data = {
                "session_id":  (None, self.session_id),
                "chunk_index": (None, str(chunk_index)),
                "device_id":   (None, device_id),
            }
            file_fields = {
                "video":      ("video.mp4",   open(files["video"],      "rb"), "video/mp4"),
                "audio":      ("audio.wav",   open(files["audio"],      "rb"), "audio/wav"),
                "imu":        ("imu.csv",     open(files["imu"],        "rb"), "text/csv"),
                "gps":        ("gps.csv",     open(files["gps"],        "rb"), "text/csv"),
                "chunk_meta": ("meta.json",   open(files["chunk_meta"], "rb"), "application/json"),
            }
            open_files = [v[1] for v in file_fields.values()]

            resp = requests.post(
                url,
                data=multipart_data,
                files=file_fields,
                headers=headers,
                timeout=60,
            )
            return resp.status_code == 200

        finally:
            for fh in open_files:
                try:
                    fh.close()
                except Exception:
                    pass

    # ── File path helpers ──────────────────────────────────────────────────────

    def _chunk_file_paths(self, chunk_index: int) -> Dict[str, Path]:
        """Return a dict of {role: Path} for all files in a chunk."""
        n = chunk_index
        return {
            "video":      self.data_dir / f"video_{n:04d}.mp4",
            "audio":      self.data_dir / f"audio_{n:04d}.wav",
            "imu":        self.data_dir / f"imu_{n:04d}.csv",
            "gps":        self.data_dir / f"gps_{n:04d}.csv",
            "chunk_meta": self.data_dir / f"chunk_meta_{n:04d}.json",
        }

    # ── SQLite queue ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the upload queue table if it does not exist."""
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS upload_queue (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    TEXT NOT NULL,
                    chunk_index   INTEGER NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    enqueued_at   REAL NOT NULL,
                    updated_at    REAL NOT NULL,
                    UNIQUE (session_id, chunk_index)
                )
            """)
            conn.commit()
            conn.close()

    def _enqueue_chunk(self, chunk_index: int, files: Dict[str, Path]) -> None:
        now = time.time()
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO upload_queue
                        (session_id, chunk_index, status, enqueued_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (self.session_id, chunk_index, _STATUS_PENDING, now, now),
                )
                conn.commit()
            finally:
                conn.close()

    def _fetch_pending_chunks(self) -> List[Dict]:
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(
                    """
                    SELECT chunk_index FROM upload_queue
                    WHERE session_id = ? AND status IN (?, ?)
                    ORDER BY chunk_index ASC
                    """,
                    (self.session_id, _STATUS_PENDING, _STATUS_UPLOADING),
                )
                rows = [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()
        return rows

    def _get_known_indices(self) -> List[int]:
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.execute(
                    "SELECT chunk_index FROM upload_queue WHERE session_id = ?",
                    (self.session_id,),
                )
                indices = [row[0] for row in cur.fetchall()]
            finally:
                conn.close()
        return indices

    def _count_pending(self) -> int:
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM upload_queue WHERE session_id = ? AND status = ?",
                    (self.session_id, _STATUS_PENDING),
                )
                return cur.fetchone()[0]
            finally:
                conn.close()

    def _update_status(self, chunk_index: int, status: str) -> None:
        now = time.time()
        with self._db_lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "UPDATE upload_queue SET status = ?, updated_at = ? WHERE session_id = ? AND chunk_index = ?",
                    (status, now, self.session_id, chunk_index),
                )
                conn.commit()
            finally:
                conn.close()

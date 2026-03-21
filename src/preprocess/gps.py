"""
GPSPreprocessor: Cleans, smooths, and enriches GPS trajectory data.

Pipeline:
1. load_gps()            — read raw 1 Hz CSV
2. kalman_smooth()       — remove jitter with a constant-velocity Kalman filter
3. compute_derived()     — distance, speed, heading, stationarity, segment IDs
4. tag_places()          — tile-based or Nominatim reverse geocoding
5. interpolate_to_fps()  — upsample 1 Hz → 30 fps for per-frame embedding
6. to_embedding_array()  — (T, 6) float32 array for model input
7. save_processed() / load_processed()  — Parquet I/O
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Haversine ─────────────────────────────────────────────────────────────────

_EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1, phi2   = math.radians(lat1), math.radians(lat2)
    dlambda      = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ── GPSPreprocessor ───────────────────────────────────────────────────────────

class GPSPreprocessor:
    """
    Preprocesses raw 1-Hz GPS CSV data into clean, feature-rich DataFrames.

    Parameters
    ----------
    kalman_process_noise : float
        Process noise (Q) for the Kalman filter.  Larger values → track faster
        changes but less smoothing.
    kalman_measurement_noise : float
        Measurement noise (R) for the Kalman filter.  Larger values → more
        smoothing but more lag.
    stationary_speed_threshold_kmh : float
        Speed below which a sample is considered stationary.
    stationary_min_duration_sec : float
        Minimum duration of slow speed for a stationary label to be applied.
    """

    def __init__(
        self,
        kalman_process_noise: float = 1e-4,
        kalman_measurement_noise: float = 1e-2,
        stationary_speed_threshold_kmh: float = 2.0,
        stationary_min_duration_sec: float = 5.0,
    ) -> None:
        self.kalman_process_noise             = kalman_process_noise
        self.kalman_measurement_noise         = kalman_measurement_noise
        self.stationary_speed_threshold_kmh   = stationary_speed_threshold_kmh
        self.stationary_min_duration_sec      = stationary_min_duration_sec

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_gps(self, csv_path: str) -> pd.DataFrame:
        """
        Load a raw GPS CSV file.

        Expected columns (all optional except ``timestamp``, ``lat``, ``lon``):
            timestamp, lat, lon, alt, speed_kmh, heading,
            fix_quality, num_satellites

        Parameters
        ----------
        csv_path : str

        Returns
        -------
        pd.DataFrame sorted by timestamp.
        """
        df = pd.read_csv(csv_path)

        required = {"timestamp", "lat", "lon"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"GPS CSV missing required columns: {missing}")

        # Fill optional columns with defaults
        defaults = {
            "alt":            0.0,
            "speed_kmh":      0.0,
            "heading":        0.0,
            "fix_quality":    1,
            "num_satellites": 0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.debug("Loaded %d GPS rows from %s", len(df), csv_path)
        return df

    # ── Kalman filter ─────────────────────────────────────────────────────────

    def kalman_smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a 2-D constant-velocity Kalman filter to latitude and longitude.

        State vector: [lat, lon, vel_lat, vel_lon]
        Observation:  [lat, lon]

        The filter is implemented from scratch with NumPy — no external library.

        Parameters
        ----------
        df : pd.DataFrame  with columns lat, lon, timestamp

        Returns
        -------
        pd.DataFrame  with lat and lon replaced by smoothed values.
            Original raw values preserved as raw_lat, raw_lon.
        """
        if len(df) < 2:
            return df.copy()

        df = df.copy()
        df["raw_lat"] = df["lat"]
        df["raw_lon"] = df["lon"]

        n = len(df)
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)

        # ── State and covariance initialisation ───────────────────────────────
        # x = [lat, lon, vel_lat, vel_lon]
        x = np.array([df["lat"].iloc[0], df["lon"].iloc[0], 0.0, 0.0], dtype=np.float64)
        P = np.eye(4, dtype=np.float64) * 1e-4

        # Measurement matrix: observe lat and lon only
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float64)
        # Measurement noise covariance
        R = np.eye(2, dtype=np.float64) * self.kalman_measurement_noise

        smoothed_lat = np.empty(n, dtype=np.float64)
        smoothed_lon = np.empty(n, dtype=np.float64)
        smoothed_lat[0] = x[0]
        smoothed_lon[0] = x[1]

        for i in range(1, n):
            dt = float(timestamps[i] - timestamps[i - 1])
            if dt <= 0:
                dt = 1.0

            # ── Transition matrix F (constant-velocity) ───────────────────────
            F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1,  0],
                          [0, 0, 0,  1]], dtype=np.float64)

            # ── Process noise Q (continuous Wiener process) ────────────────────
            q = self.kalman_process_noise
            dt2  = dt * dt
            dt3  = dt2 * dt
            dt4  = dt3 * dt
            Q = np.array([
                [dt4 / 4, 0,       dt3 / 2, 0      ],
                [0,       dt4 / 4, 0,       dt3 / 2],
                [dt3 / 2, 0,       dt2,     0      ],
                [0,       dt3 / 2, 0,       dt2    ],
            ], dtype=np.float64) * q

            # ── Predict ──────────────────────────────────────────────────────
            x = F @ x
            P = F @ P @ F.T + Q

            # ── Update ───────────────────────────────────────────────────────
            z = np.array([df["lat"].iloc[i], df["lon"].iloc[i]], dtype=np.float64)
            y  = z - H @ x                     # innovation
            S  = H @ P @ H.T + R               # innovation covariance
            K  = P @ H.T @ np.linalg.inv(S)    # Kalman gain
            x  = x + K @ y
            P  = (np.eye(4, dtype=np.float64) - K @ H) @ P

            smoothed_lat[i] = x[0]
            smoothed_lon[i] = x[1]

        df["lat"] = smoothed_lat
        df["lon"] = smoothed_lon
        logger.debug("Kalman smoothing applied to %d GPS points", n)
        return df

    # ── Derived features ──────────────────────────────────────────────────────

    def compute_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived columns to a GPS DataFrame.

        Added columns
        -------------
        distance_m          : cumulative distance travelled in metres
        speed_kmh_computed  : speed computed from position deltas
        heading_computed    : bearing between consecutive points (0-360°)
        is_stationary       : True if slow for >= stationary_min_duration_sec
        segment_id          : increments on each stationary→moving transition

        Parameters
        ----------
        df : pd.DataFrame  (must have lat, lon, timestamp)

        Returns
        -------
        pd.DataFrame  with new columns appended.
        """
        df = df.copy()
        n  = len(df)

        lats = df["lat"].to_numpy(dtype=np.float64)
        lons = df["lon"].to_numpy(dtype=np.float64)
        ts   = df["timestamp"].to_numpy(dtype=np.float64)

        # ── Distance and speed ────────────────────────────────────────────────
        dist_deltas    = np.zeros(n, dtype=np.float64)
        speed_computed = np.zeros(n, dtype=np.float64)
        heading_computed = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            d_m  = _haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])
            dt_s = max(ts[i] - ts[i - 1], 1e-6)

            dist_deltas[i]    = d_m
            speed_computed[i] = (d_m / dt_s) * 3.6  # m/s → km/h
            heading_computed[i] = _bearing_deg(lats[i - 1], lons[i - 1], lats[i], lons[i])

        # Carry first heading from second sample
        if n > 1:
            heading_computed[0] = heading_computed[1]

        df["distance_m"]          = np.cumsum(dist_deltas)
        df["speed_kmh_computed"]  = speed_computed
        df["heading_computed"]    = heading_computed

        # ── Stationarity ──────────────────────────────────────────────────────
        slow_mask = speed_computed < self.stationary_speed_threshold_kmh

        # Apply minimum-duration requirement
        is_stationary = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if slow_mask[i]:
                # Find end of slow run
                j = i
                while j < n and slow_mask[j]:
                    j += 1
                duration = ts[j - 1] - ts[i] if j > i else 0.0
                if duration >= self.stationary_min_duration_sec:
                    is_stationary[i:j] = True
                i = j
            else:
                i += 1

        df["is_stationary"] = is_stationary

        # ── Segment ID ────────────────────────────────────────────────────────
        # Increment segment_id on every stationary→moving transition
        segment_ids = np.zeros(n, dtype=np.int32)
        seg = 0
        was_stationary = bool(is_stationary[0]) if n > 0 else False
        for i in range(n):
            if was_stationary and not is_stationary[i]:
                seg += 1
            segment_ids[i] = seg
            was_stationary = bool(is_stationary[i])

        df["segment_id"] = segment_ids
        return df

    # ── Place tagging ─────────────────────────────────────────────────────────

    def tag_places(
        self,
        df: pd.DataFrame,
        nominatim_offline: bool = True,
    ) -> pd.DataFrame:
        """
        Add ``place_name`` and ``place_type`` columns to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame  (must have lat, lon columns)
        nominatim_offline : bool
            If True, assign tile IDs based on rounded lat/lon (no network).
            If False, call the Nominatim REST API at 1 req/sec rate limit.

        Returns
        -------
        pd.DataFrame  with place_name and place_type columns.
        """
        df = df.copy()

        if nominatim_offline:
            df["place_name"] = df.apply(
                lambda row: f"tile_{row['lat']:.3f}_{row['lon']:.3f}", axis=1
            )
            df["place_type"] = "tile"
        else:
            df["place_name"] = ""
            df["place_type"] = ""
            try:
                import urllib.request as _ur
                import urllib.parse  as _up
            except ImportError:
                logger.warning("urllib not available; place tagging skipped")
                return df

            last_req_time = 0.0
            cache: Dict[Tuple[float, float], Tuple[str, str]] = {}

            for idx_row, row in df.iterrows():
                lat_r = round(float(row["lat"]), 4)
                lon_r = round(float(row["lon"]), 4)
                key   = (lat_r, lon_r)

                if key in cache:
                    place_name, place_type = cache[key]
                else:
                    # Rate-limit: 1 request per second
                    elapsed = time.time() - last_req_time
                    if elapsed < 1.0:
                        time.sleep(1.0 - elapsed)

                    url = (
                        "https://nominatim.openstreetmap.org/reverse"
                        f"?lat={lat_r}&lon={lon_r}&format=json"
                    )
                    try:
                        req = _ur.Request(url, headers={"User-Agent": "vla-gps-preprocessor/1.0"})
                        with _ur.urlopen(req, timeout=5) as resp:
                            data = __import__("json").loads(resp.read())
                        last_req_time = time.time()

                        address = data.get("address", {})
                        place_name = (
                            address.get("neighbourhood")
                            or address.get("suburb")
                            or address.get("city_district")
                            or address.get("city")
                            or data.get("display_name", "")[:60]
                        )
                        # Determine place_type from OSM class
                        place_type = data.get("class", "unknown")
                    except Exception as exc:
                        logger.warning("Nominatim request failed for %s: %s", key, exc)
                        place_name = f"tile_{lat_r:.3f}_{lon_r:.3f}"
                        place_type = "unknown"
                        last_req_time = time.time()

                    cache[key] = (place_name, place_type)

                df.at[idx_row, "place_name"] = place_name
                df.at[idx_row, "place_type"] = place_type

        return df

    # ── Upsampling ────────────────────────────────────────────────────────────

    def interpolate_to_fps(self, df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
        """
        Linearly interpolate the 1-Hz GPS track to video frame rate.

        Columns interpolated: lat, lon, alt, speed_kmh, heading.
        Categorical columns (is_stationary, segment_id, fix_quality,
        num_satellites, place_name, place_type) are forward-filled.

        Parameters
        ----------
        df : pd.DataFrame  with timestamp column
        fps : float  target frame rate (default 30)

        Returns
        -------
        pd.DataFrame  at ~fps samples per second.
        """
        if df.empty:
            return df.copy()

        ts_orig  = df["timestamp"].to_numpy(dtype=np.float64)
        t_start  = ts_orig[0]
        t_end    = ts_orig[-1]
        dt       = 1.0 / fps

        # New timestamp grid at target fps
        ts_new = np.arange(t_start, t_end + dt, dt, dtype=np.float64)

        numeric_cols = ["lat", "lon", "alt", "speed_kmh", "heading",
                        "speed_kmh_computed", "heading_computed",
                        "distance_m"]

        result = pd.DataFrame({"timestamp": ts_new})

        for col in numeric_cols:
            if col in df.columns:
                vals = df[col].to_numpy(dtype=np.float64)
                result[col] = np.interp(ts_new, ts_orig, vals)

        # Forward-fill categorical columns
        ffill_cols = ["is_stationary", "segment_id", "fix_quality",
                      "num_satellites", "place_name", "place_type"]
        for col in ffill_cols:
            if col not in df.columns:
                continue
            indices = np.searchsorted(ts_orig, ts_new, side="right") - 1
            indices = np.clip(indices, 0, len(df) - 1)
            result[col] = df[col].to_numpy()[indices]

        logger.debug(
            "GPS interpolated: %d → %d rows (%.1f fps)",
            len(df), len(result), fps,
        )
        return result

    # ── Embedding array ───────────────────────────────────────────────────────

    def to_embedding_array(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert the GPS DataFrame to a model-ready float32 array.

        Output shape: ``(T, 6)`` with columns:
            0: lat_norm       — latitude normalised to [-1, 1] over the session
            1: lon_norm       — longitude normalised to [-1, 1] over the session
            2: alt_norm       — altitude normalised to [0, 1]
            3: speed_norm     — speed (km/h) normalised to [0, 1] clipped at 120 km/h
            4: heading_sin    — sin(heading_rad)  cyclic encoding
            5: heading_cos    — cos(heading_rad)  cyclic encoding

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray  shape (T, 6), dtype float32
        """
        T = len(df)
        if T == 0:
            return np.zeros((0, 6), dtype=np.float32)

        lats      = df["lat"].to_numpy(dtype=np.float64)
        lons      = df["lon"].to_numpy(dtype=np.float64)
        alts      = df.get("alt",      pd.Series(np.zeros(T))).to_numpy(dtype=np.float64)
        speeds    = df.get("speed_kmh", pd.Series(np.zeros(T))).to_numpy(dtype=np.float64)
        headings  = df.get("heading",   pd.Series(np.zeros(T))).to_numpy(dtype=np.float64)

        # Normalise lat / lon relative to session extents
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        lat_range = max(lat_max - lat_min, 1e-9)
        lon_range = max(lon_max - lon_min, 1e-9)

        lat_norm  = 2.0 * (lats - lat_min) / lat_range - 1.0
        lon_norm  = 2.0 * (lons - lon_min) / lon_range - 1.0

        alt_min, alt_max = alts.min(), alts.max()
        alt_range = max(alt_max - alt_min, 1e-9)
        alt_norm  = (alts - alt_min) / alt_range

        speed_norm = np.clip(speeds / 120.0, 0.0, 1.0)

        heading_rad = np.radians(headings)
        heading_sin = np.sin(heading_rad)
        heading_cos = np.cos(heading_rad)

        arr = np.stack(
            [lat_norm, lon_norm, alt_norm, speed_norm, heading_sin, heading_cos],
            axis=1,
        ).astype(np.float32)

        return arr

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_processed(self, df: pd.DataFrame, output_path: str) -> None:
        """Save the processed GPS DataFrame to Parquet format.

        Parameters
        ----------
        df : pd.DataFrame
        output_path : str  path ending in ``.parquet``
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.debug("GPS parquet saved → %s (%d rows)", output_path, len(df))

    def load_processed(self, path: str) -> pd.DataFrame:
        """Load a previously saved GPS Parquet file.

        Parameters
        ----------
        path : str

        Returns
        -------
        pd.DataFrame
        """
        df = pd.read_parquet(path, engine="pyarrow")
        logger.debug("GPS parquet loaded ← %s (%d rows)", path, len(df))
        return df

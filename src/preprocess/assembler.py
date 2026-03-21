"""
ChunkAssembler: Concatenates per-chunk files into a single assembled session.

Runs on the server after all 30-second chunks have arrived.  Handles missing
chunks gracefully (logs warnings, skips gaps), validates temporal continuity,
and writes assembled outputs to ``session_dir/assembled/``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# scipy is used for WAV concatenation
try:
    from scipy.io import wavfile as _wavfile
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False
    logger.warning("scipy not installed; audio assembly will be skipped.")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AssembledSession:
    """Result of a successful session assembly pass."""
    session_id: str
    video_path: str
    audio_path: str
    imu_path: str
    gps_path: str
    metadata: Dict
    total_duration_sec: float
    chunk_count: int


# ── ChunkAssembler ─────────────────────────────────────────────────────────────

class ChunkAssembler:
    """
    Discovers and concatenates all chunk files for a session.

    Chunk files follow the naming convention::

        video_NNNN.mp4
        audio_NNNN.wav
        imu_NNNN.csv
        gps_NNNN.csv
        chunk_meta_NNNN.json

    where NNNN is a zero-padded four-digit chunk index.

    Parameters
    ----------
    video_fps : int
        Nominal video frame-rate; used to estimate total frame count.
    chunk_duration_sec : float
        Expected duration of each chunk (default 30 s) for gap checks.
    max_timestamp_gap_sec : float
        Warn if consecutive chunk timestamps are further apart than this.
    """

    def __init__(
        self,
        video_fps: int = 30,
        chunk_duration_sec: float = 30.0,
        max_timestamp_gap_sec: float = 5.0,
    ) -> None:
        self.video_fps = video_fps
        self.chunk_duration_sec = chunk_duration_sec
        self.max_timestamp_gap_sec = max_timestamp_gap_sec

    # ── Public API ──────────────────────────────────────────────────────────────

    def assemble_session(self, session_dir: str) -> AssembledSession:
        """
        Assemble all chunk files found in *session_dir*.

        Parameters
        ----------
        session_dir : str
            Top-level directory that contains the raw chunk files and
            ``chunk_meta_NNNN.json`` files.

        Returns
        -------
        AssembledSession
        """
        base = Path(session_dir)
        out_dir = base / "assembled"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Discover chunks
        chunk_indices = self._discover_chunk_indices(base)
        if not chunk_indices:
            raise FileNotFoundError(f"No chunks found in {session_dir}")

        self._validate_chunk_sequence(chunk_indices)
        logger.info("Assembling %d chunks from %s", len(chunk_indices), session_dir)

        # Load all chunk metadata
        all_meta = self._load_chunk_metas(base, chunk_indices)
        session_id = self._extract_session_id(all_meta, base)

        # Assemble each modality
        video_path = self._assemble_video(base, chunk_indices, out_dir)
        audio_path = self._assemble_audio(base, chunk_indices, out_dir)
        imu_path   = self._assemble_csv(base, chunk_indices, "imu",    out_dir)
        gps_path   = self._assemble_csv(base, chunk_indices, "gps",    out_dir)

        # Temporal continuity check
        self._check_temporal_continuity(base, chunk_indices)

        # Compute total duration
        total_duration = len(chunk_indices) * self.chunk_duration_sec
        # If metadata has precise timing use that
        if all_meta:
            starts = [m.get("chunk_start_time", 0.0) for m in all_meta if "chunk_start_time" in m]
            ends   = [m.get("chunk_end_time", 0.0)   for m in all_meta if "chunk_end_time"   in m]
            if starts and ends:
                total_duration = max(ends) - min(starts)

        # Build combined metadata
        metadata = {
            "session_id":        session_id,
            "chunk_count":       len(chunk_indices),
            "chunk_indices":     chunk_indices,
            "total_duration_sec": total_duration,
            "assembled_at":      pd.Timestamp.now().isoformat(),
            "chunk_metas":       all_meta,
        }
        meta_path = out_dir / "session_meta.json"
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info(
            "Assembly complete: %.1f sec, %d chunks → %s",
            total_duration, len(chunk_indices), out_dir,
        )

        return AssembledSession(
            session_id=session_id,
            video_path=str(video_path),
            audio_path=str(audio_path),
            imu_path=str(imu_path),
            gps_path=str(gps_path),
            metadata=metadata,
            total_duration_sec=total_duration,
            chunk_count=len(chunk_indices),
        )

    def get_assembly_stats(self, session_dir: str) -> Dict:
        """
        Return statistics about chunk coverage without performing assembly.

        Returns
        -------
        dict with keys:
            chunks_found, chunks_missing, coverage_pct,
            gaps, total_frames_estimate, total_duration_estimate_sec
        """
        base = Path(session_dir)
        chunk_indices = self._discover_chunk_indices(base)

        if not chunk_indices:
            return {
                "chunks_found": 0,
                "chunks_missing": 0,
                "coverage_pct": 0.0,
                "gaps": [],
                "total_frames_estimate": 0,
                "total_duration_estimate_sec": 0.0,
            }

        expected = list(range(chunk_indices[0], chunk_indices[-1] + 1))
        missing  = [i for i in expected if i not in set(chunk_indices)]
        gaps: List[Dict] = []
        for i, idx in enumerate(chunk_indices[:-1]):
            next_idx = chunk_indices[i + 1]
            if next_idx - idx > 1:
                gaps.append({"after_chunk": idx, "missing_count": next_idx - idx - 1})

        coverage = len(chunk_indices) / len(expected) * 100 if expected else 0.0
        total_frames = len(chunk_indices) * self.chunk_duration_sec * self.video_fps
        total_duration = len(chunk_indices) * self.chunk_duration_sec

        return {
            "chunks_found": len(chunk_indices),
            "chunks_missing": len(missing),
            "coverage_pct": round(coverage, 2),
            "gaps": gaps,
            "total_frames_estimate": int(total_frames),
            "total_duration_estimate_sec": round(total_duration, 1),
        }

    # ── Discovery ───────────────────────────────────────────────────────────────

    def _discover_chunk_indices(self, base: Path) -> List[int]:
        """Return sorted list of chunk indices present in base directory."""
        indices: List[int] = []

        # Also check base/chunks/ subdirectory (used by ingest API)
        search_dirs = [base]
        chunks_subdir = base / "chunks"
        if chunks_subdir.exists():
            # Chunks stored as base/chunks/chunk_NNNN/video_NNNN.mp4
            for subdir in sorted(chunks_subdir.iterdir()):
                if subdir.is_dir() and subdir.name.startswith("chunk_"):
                    try:
                        idx = int(subdir.name.split("_", 1)[1])
                        # Verify at least the video file exists
                        if any(subdir.glob(f"video_{idx:04d}.mp4")):
                            indices.append(idx)
                    except (ValueError, IndexError):
                        pass
            if indices:
                return sorted(indices)

        # Flat layout: base/video_NNNN.mp4
        for p in sorted(base.glob("video_*.mp4")):
            stem = p.stem  # "video_0000"
            try:
                idx = int(stem.split("_")[-1])
                indices.append(idx)
            except ValueError:
                pass
        return sorted(indices)

    def _get_chunk_paths(self, base: Path, chunk_index: int) -> Dict[str, Path]:
        """
        Resolve file paths for a given chunk index.

        Handles both flat layout (base/video_NNNN.mp4) and nested layout
        (base/chunks/chunk_NNNN/video_NNNN.mp4).
        """
        n = chunk_index
        nested = base / "chunks" / f"chunk_{n:04d}"
        if nested.exists():
            root = nested
        else:
            root = base

        return {
            "video":      root / f"video_{n:04d}.mp4",
            "audio":      root / f"audio_{n:04d}.wav",
            "imu":        root / f"imu_{n:04d}.csv",
            "gps":        root / f"gps_{n:04d}.csv",
            "chunk_meta": root / f"chunk_meta_{n:04d}.json",
        }

    # ── Validation ──────────────────────────────────────────────────────────────

    def _validate_chunk_sequence(self, indices: List[int]) -> None:
        """Log warnings for any gaps in the chunk sequence."""
        for i, idx in enumerate(indices[:-1]):
            next_idx = indices[i + 1]
            if next_idx - idx > 1:
                missing = list(range(idx + 1, next_idx))
                logger.warning(
                    "Chunk sequence gap: chunks %s are missing (between %d and %d)",
                    missing, idx, next_idx,
                )

    def _check_temporal_continuity(self, base: Path, chunk_indices: List[int]) -> None:
        """
        Warn if the timestamp gap between consecutive chunks exceeds the threshold.
        """
        prev_end: Optional[float] = None
        for idx in chunk_indices:
            paths = self._get_chunk_paths(base, idx)
            meta_path = paths["chunk_meta"]
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
                chunk_start = float(meta.get("chunk_start_time", 0))
                chunk_end   = float(meta.get("chunk_end_time", chunk_start + self.chunk_duration_sec))
            except Exception:
                continue

            if prev_end is not None:
                gap = chunk_start - prev_end
                if gap > self.max_timestamp_gap_sec:
                    logger.warning(
                        "Temporal gap of %.1f s before chunk %d (expected < %.1f s)",
                        gap, idx, self.max_timestamp_gap_sec,
                    )
            prev_end = chunk_end

    # ── Video assembly ──────────────────────────────────────────────────────────

    def _assemble_video(self, base: Path, chunk_indices: List[int], out_dir: Path) -> Path:
        """Concatenate MP4 files using ffmpeg concat demuxer."""
        out_path = out_dir / "video.mp4"

        video_paths: List[Path] = []
        for idx in chunk_indices:
            p = self._get_chunk_paths(base, idx)["video"]
            if p.exists():
                video_paths.append(p)
            else:
                logger.warning("Video file missing for chunk %d: %s", idx, p)

        if not video_paths:
            logger.error("No video files found; skipping video assembly")
            return out_path

        # Write concat list file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            for p in video_paths:
                # ffmpeg requires forward-slash paths on all platforms
                tmp.write(f"file '{p.as_posix()}'\n")
            concat_list = tmp.name

        try:
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                str(out_path),
            ]
            logger.debug("Running: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(
                    "ffmpeg failed (rc=%d): %s",
                    result.returncode, result.stderr.decode(errors="replace")[-500:],
                )
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out during video assembly")
        except FileNotFoundError:
            logger.error("ffmpeg not found; video assembly skipped")
        finally:
            try:
                os.unlink(concat_list)
            except OSError:
                pass

        return out_path

    # ── Audio assembly ──────────────────────────────────────────────────────────

    def _assemble_audio(self, base: Path, chunk_indices: List[int], out_dir: Path) -> Path:
        """Concatenate WAV files by reading PCM data with scipy and stacking."""
        out_path = out_dir / "audio.wav"

        if not _HAS_SCIPY:
            logger.error("scipy not available; audio assembly skipped")
            return out_path

        all_samples: List[np.ndarray] = []
        sample_rate: Optional[int] = None

        for idx in chunk_indices:
            p = self._get_chunk_paths(base, idx)["audio"]
            if not p.exists():
                logger.warning("Audio file missing for chunk %d: %s", idx, p)
                continue
            try:
                sr, data = _wavfile.read(str(p))
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning(
                        "Chunk %d sample rate %d != expected %d; skipping",
                        idx, sr, sample_rate,
                    )
                    continue

                # Ensure mono float32
                if data.ndim > 1:
                    data = data.mean(axis=1)
                all_samples.append(data.astype(np.float32))

            except Exception as exc:
                logger.error("Cannot read audio chunk %d: %s", idx, exc)

        if not all_samples or sample_rate is None:
            logger.error("No audio data assembled")
            return out_path

        combined = np.concatenate(all_samples, axis=0)

        # Normalize to int16 range for output
        max_val = np.abs(combined).max()
        if max_val > 0:
            combined = combined / max_val

        combined_int16 = (combined * 32767).astype(np.int16)
        _wavfile.write(str(out_path), sample_rate, combined_int16)
        logger.debug("Audio assembled: %.1f sec, %d samples", len(combined) / sample_rate, len(combined))
        return out_path

    # ── CSV assembly (IMU + GPS) ────────────────────────────────────────────────

    def _assemble_csv(
        self,
        base: Path,
        chunk_indices: List[int],
        modality: str,
        out_dir: Path,
    ) -> Path:
        """
        Concatenate CSV chunks (IMU or GPS) into a single sorted, deduplicated file.
        """
        out_path = out_dir / f"{modality}.csv"
        frames: List[pd.DataFrame] = []

        for idx in chunk_indices:
            csv_path = self._get_chunk_paths(base, idx)[modality]
            if not csv_path.exists():
                logger.warning("%s CSV missing for chunk %d: %s", modality.upper(), idx, csv_path)
                continue
            try:
                df = pd.read_csv(csv_path)
                frames.append(df)
            except Exception as exc:
                logger.error("Cannot read %s CSV chunk %d: %s", modality, idx, exc)

        if not frames:
            logger.warning("No %s CSV data found; writing empty file", modality)
            out_path.write_text("")
            return out_path

        combined = pd.concat(frames, ignore_index=True)

        if "timestamp" in combined.columns:
            combined = combined.sort_values("timestamp")
            combined = combined.drop_duplicates(subset=["timestamp"], keep="first")

        combined.to_csv(str(out_path), index=False)
        logger.debug(
            "%s CSV assembled: %d rows → %s", modality.upper(), len(combined), out_path
        )
        return out_path

    # ── Metadata helpers ────────────────────────────────────────────────────────

    def _load_chunk_metas(self, base: Path, chunk_indices: List[int]) -> List[Dict]:
        """Load and return all chunk_meta JSON files."""
        metas: List[Dict] = []
        for idx in chunk_indices:
            p = self._get_chunk_paths(base, idx)["chunk_meta"]
            if not p.exists():
                continue
            try:
                with open(p) as fh:
                    metas.append(json.load(fh))
            except Exception as exc:
                logger.warning("Cannot read chunk meta %d: %s", idx, exc)
        return metas

    def _extract_session_id(self, metas: List[Dict], base: Path) -> str:
        """Extract session_id from chunk metas, falling back to directory name."""
        for m in metas:
            sid = m.get("session_id")
            if sid:
                return str(sid)
        return base.name

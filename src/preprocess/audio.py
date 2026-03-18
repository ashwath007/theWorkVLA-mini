"""
AudioPreprocessor: Load, resample, normalize, and chunk audio aligned to video frames.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

_TARGET_SR = 16000


class AudioPreprocessor:
    """
    Audio preprocessing utilities for VLA data pipeline.

    All methods work on raw PCM float32 arrays (values in [-1, 1]).
    """

    # ── Load ────────────────────────────────────────────────────────────────────

    def load_audio(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """
        Load a WAV file and return mono float32 PCM samples.

        Parameters
        ----------
        wav_path : str
            Path to .wav file (any bit-depth supported).

        Returns
        -------
        samples : np.ndarray  shape (N,) float32 in [-1.0, 1.0]
        sample_rate : int
        """
        try:
            with wave.open(wav_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth  = wf.getsampwidth()
                framerate  = wf.getframerate()
                n_frames   = wf.getnframes()
                raw        = wf.readframes(n_frames)
        except Exception as exc:
            raise IOError(f"Cannot read WAV file '{wav_path}': {exc}") from exc

        # Decode bytes → numpy
        if sampwidth == 1:
            dtype = np.uint8
            max_val = 128.0
        elif sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sampwidth == 3:
            # 24-bit: unpack manually
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            pcm_int = (arr[:, 0].astype(np.int32) |
                       (arr[:, 1].astype(np.int32) << 8) |
                       (arr[:, 2].astype(np.int32) << 16))
            pcm_int[pcm_int >= (1 << 23)] -= (1 << 24)
            samples = pcm_int.astype(np.float32) / float(1 << 23)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return samples, framerate
        elif sampwidth == 4:
            dtype = np.int32
            max_val = 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if sampwidth == 1:
            pcm = (pcm - 128.0) / 128.0
        else:
            pcm = pcm / max_val

        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)

        return pcm, framerate

    # ── Resample ─────────────────────────────────────────────────────────────────

    def resample_to_16k(
        self,
        samples: np.ndarray,
        orig_sr: int,
        target_sr: int = _TARGET_SR,
    ) -> np.ndarray:
        """
        Resample audio from orig_sr to target_sr using polyphase resampling.

        Parameters
        ----------
        samples : np.ndarray  float32
        orig_sr : int         original sample rate
        target_sr : int       target sample rate (default 16000)

        Returns
        -------
        np.ndarray  float32 resampled audio.
        """
        if orig_sr == target_sr:
            return samples.copy()

        from math import gcd
        g = gcd(orig_sr, target_sr)
        up   = target_sr // g
        down = orig_sr   // g

        resampled = scipy_signal.resample_poly(samples, up, down)
        return resampled.astype(np.float32)

    # ── Normalize ────────────────────────────────────────────────────────────────

    def normalize_audio(self, samples: np.ndarray) -> np.ndarray:
        """
        Peak-normalize audio to [-1, 1].

        Parameters
        ----------
        samples : np.ndarray  float32

        Returns
        -------
        np.ndarray  float32 peak-normalized.
        """
        peak = np.max(np.abs(samples))
        if peak < 1e-8:
            return samples.copy()
        return (samples / peak).astype(np.float32)

    # ── Chunk ─────────────────────────────────────────────────────────────────────

    def chunk_by_timestamps(
        self,
        samples: np.ndarray,
        sr: int,
        timestamps: np.ndarray,
        window_sec: Optional[float] = None,
        audio_start_time: float = 0.0,
    ) -> List[np.ndarray]:
        """
        Slice audio into chunks aligned to video frame timestamps.

        For each timestamp t[i], the chunk spans [t[i], t[i+1]) (or a fixed
        window_sec if provided).  All chunks are zero-padded to the same length.

        Parameters
        ----------
        samples : np.ndarray  float32 mono audio
        sr : int              sample rate of `samples`
        timestamps : np.ndarray  shape (T,) float64 — absolute Unix epoch or
                     relative seconds.  Must be monotonically increasing.
        window_sec : float, optional
            Fixed window width in seconds.  If None, uses the inter-frame interval.
        audio_start_time : float
            Time (epoch or relative) of the first audio sample.

        Returns
        -------
        list of np.ndarray, length T, each shape (samples_per_chunk,).
        """
        if len(timestamps) == 0:
            return []

        # Default window = median inter-frame gap
        if window_sec is None:
            if len(timestamps) > 1:
                window_sec = float(np.median(np.diff(timestamps)))
            else:
                window_sec = 1.0 / 30

        samples_per_chunk = max(1, int(window_sec * sr))
        chunks: List[np.ndarray] = []

        for t in timestamps:
            # Convert absolute timestamp → sample index
            offset_sec = float(t) - audio_start_time
            idx_start  = int(offset_sec * sr)
            idx_end    = idx_start + samples_per_chunk

            chunk = np.zeros(samples_per_chunk, dtype=np.float32)
            src = samples[max(0, idx_start):min(len(samples), idx_end)]
            n   = min(len(src), samples_per_chunk)
            chunk[:n] = src[:n]
            chunks.append(chunk)

        return chunks

    # ── Convenience pipeline ─────────────────────────────────────────────────────

    def preprocess(
        self,
        wav_path: str,
        timestamps: np.ndarray,
        audio_start_time: float = 0.0,
        normalize: bool = True,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Full preprocessing: load → resample 16kHz → normalize → chunk by timestamps.

        Returns
        -------
        chunks : list of np.ndarray
        sample_rate : int  (always 16000)
        """
        samples, sr = self.load_audio(wav_path)
        samples = self.resample_to_16k(samples, sr)
        if normalize:
            samples = self.normalize_audio(samples)
        chunks = self.chunk_by_timestamps(samples, _TARGET_SR, timestamps,
                                          audio_start_time=audio_start_time)
        return chunks, _TARGET_SR

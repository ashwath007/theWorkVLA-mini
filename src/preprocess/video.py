"""
VideoPreprocessor: Frame extraction, resizing, normalization, and face blurring.

All face blurring uses OpenCV's Haar cascade detector for privacy compliance.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Path to OpenCV's built-in frontal face cascade
_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]


class VideoPreprocessor:
    """
    Preprocessing utilities for video frames captured from an egocentric headset.

    Parameters
    ----------
    face_cascade_path : str
        Path to Haar cascade XML for face detection. Defaults to OpenCV built-in.
    face_blur_kernel : int
        Gaussian blur kernel size applied over detected faces (must be odd).
    """

    def __init__(
        self,
        face_cascade_path: str = _FACE_CASCADE_PATH,
        face_blur_kernel: int = 99,
    ) -> None:
        self.face_blur_kernel = face_blur_kernel
        try:
            self._face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if self._face_cascade.empty():
                raise RuntimeError("Loaded cascade is empty.")
        except Exception as exc:
            logger.warning("Face cascade load failed (%s); face blurring disabled.", exc)
            self._face_cascade = None

    # ── Public API ──────────────────────────────────────────────────────────────

    def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Read all (or up to max_frames) frames from a video file.

        Parameters
        ----------
        video_path : str
            Path to the video file (e.g., video.mp4).
        max_frames : int, optional
            Limit the number of frames extracted.

        Returns
        -------
        frames : list of np.ndarray
            Each frame is BGR uint8 with shape (H, W, 3).
        timestamps : np.ndarray
            Per-frame timestamps in seconds (relative to start of video).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                timestamps.append(idx / fps)
                idx += 1
                if max_frames and idx >= max_frames:
                    break
        finally:
            cap.release()

        logger.info("Extracted %d frames from %s", len(frames), video_path)
        return frames, np.array(timestamps, dtype=np.float64)

    def resize_frames(
        self,
        frames: List[np.ndarray],
        target_size: Tuple[int, int] = (224, 224),
    ) -> List[np.ndarray]:
        """
        Resize a list of frames to target_size (width, height).

        Parameters
        ----------
        frames : list of np.ndarray
        target_size : (width, height)

        Returns
        -------
        list of np.ndarray with shape (height, width, 3) uint8.
        """
        return [cv2.resize(f, target_size, interpolation=cv2.INTER_LINEAR) for f in frames]

    def normalize_frames(
        self,
        frames: List[np.ndarray],
    ) -> np.ndarray:
        """
        Convert uint8 BGR frames to float32 RGB in [0, 1].

        Parameters
        ----------
        frames : list of np.ndarray  shape (H, W, 3) uint8

        Returns
        -------
        np.ndarray of shape (T, H, W, 3) float32.
        """
        normalized = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized.append(rgb.astype(np.float32) / 255.0)
        return np.stack(normalized, axis=0)

    def blur_faces(
        self,
        frames: List[np.ndarray],
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (30, 30),
    ) -> List[np.ndarray]:
        """
        Detect and Gaussian-blur faces in each frame for privacy protection.

        Parameters
        ----------
        frames : list of np.ndarray  BGR uint8
        scale_factor : float
            Haar cascade scale factor.
        min_neighbors : int
            Haar cascade min-neighbours (higher = fewer false positives).
        min_face_size : (int, int)
            Minimum face size in pixels.

        Returns
        -------
        list of np.ndarray with faces blurred.
        """
        if self._face_cascade is None or self._face_cascade.empty():
            logger.debug("Face cascade not available; returning frames unchanged.")
            return frames

        result: List[np.ndarray] = []
        kernel = self.face_blur_kernel | 1  # ensure odd

        for frame in frames:
            out = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_face_size,
            )
            for (x, y, w, h) in faces:
                roi = out[y:y+h, x:x+w]
                out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
            result.append(out)

        return result

    def bgr_to_rgb(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Convert a list of BGR frames to RGB (in-place copy)."""
        return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]

    def preprocess_pipeline(
        self,
        video_path: str,
        target_size: Tuple[int, int] = (224, 224),
        apply_face_blur: bool = True,
        max_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline: extract → blur faces → resize → normalize.

        Returns
        -------
        frames_array : np.ndarray  (T, H, W, 3) float32 RGB
        timestamps   : np.ndarray  (T,) float64 seconds
        """
        frames, timestamps = self.extract_frames(video_path, max_frames=max_frames)

        if apply_face_blur:
            frames = self.blur_faces(frames)

        frames = self.resize_frames(frames, target_size=target_size)
        frames_array = self.normalize_frames(frames)

        return frames_array, timestamps

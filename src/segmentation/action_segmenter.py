"""
ActionSegmenter: Detect action boundaries in egocentric video using optical flow.

Two modes:
  1. optical_flow — Lucas-Kanade sparse optical flow to detect motion magnitude changes
  2. uniform — fallback fixed-window segmentation

ActionSegment is a simple dataclass carrying segment metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lucas-Kanade parameters
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# ShiTomasi corner detection parameters (seed points for LK)
_FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7,
)


@dataclass
class ActionSegment:
    """Represents a detected action segment in a video."""
    start_frame: int
    end_frame: int
    confidence: float           # 0.0 – 1.0
    motion_magnitude: float     # mean optical flow magnitude in segment

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration_sec(self) -> float:
        return self.num_frames / 30.0   # assume 30fps; override if needed

    def to_dict(self) -> dict:
        return {
            "start_frame":       self.start_frame,
            "end_frame":         self.end_frame,
            "confidence":        round(self.confidence, 4),
            "motion_magnitude":  round(self.motion_magnitude, 4),
        }


class ActionSegmenter:
    """
    Detects action boundaries in egocentric video.

    Parameters
    ----------
    flow_threshold_low : float
        Frames with mean flow magnitude below this are considered "idle".
    flow_threshold_high : float
        Frames above this are "high motion" — boundaries are detected when
        magnitude exceeds this after a quiet period.
    boundary_min_gap : int
        Minimum frames between two boundaries to avoid over-segmentation.
    fps : int
        Assumed video frames per second (used for duration calculations).
    """

    def __init__(
        self,
        flow_threshold_low: float = 1.0,
        flow_threshold_high: float = 3.0,
        boundary_min_gap: int = 15,
        fps: int = 30,
    ) -> None:
        self.flow_threshold_low  = flow_threshold_low
        self.flow_threshold_high = flow_threshold_high
        self.boundary_min_gap    = boundary_min_gap
        self.fps                 = fps

    # ── Public API ──────────────────────────────────────────────────────────────

    def segment_optical_flow(
        self,
        frames: List[np.ndarray],
    ) -> List[ActionSegment]:
        """
        Detect action segments using Lucas-Kanade sparse optical flow.

        An action segment starts when motion magnitude exceeds
        flow_threshold_high and ends when it drops below flow_threshold_low
        (with hysteresis).

        Parameters
        ----------
        frames : list of np.ndarray  BGR uint8

        Returns
        -------
        list of ActionSegment
        """
        if len(frames) < 2:
            logger.warning("Not enough frames for optical flow segmentation.")
            return [ActionSegment(0, len(frames), 0.5, 0.0)] if frames else []

        flow_magnitudes = self._compute_flow_magnitudes(frames)
        segments        = self._threshold_to_segments(flow_magnitudes)
        segments        = self.merge_short_segments(segments, min_frames=self.boundary_min_gap)

        logger.info(
            "Optical flow segmentation: %d segments from %d frames",
            len(segments), len(frames),
        )
        return segments

    def segment_uniform(
        self,
        frames: List[np.ndarray],
        window_sec: float = 3.0,
        fps: Optional[int] = None,
    ) -> List[ActionSegment]:
        """
        Segment video into uniform fixed-width windows.

        Parameters
        ----------
        frames : list of np.ndarray
        window_sec : float  duration of each segment in seconds
        fps : int  frames-per-second (defaults to self.fps)

        Returns
        -------
        list of ActionSegment
        """
        fps = fps or self.fps
        window_frames = max(1, int(window_sec * fps))
        T = len(frames)

        segments: List[ActionSegment] = []
        start = 0
        while start < T:
            end = min(start + window_frames, T)
            segments.append(ActionSegment(
                start_frame=start,
                end_frame=end,
                confidence=1.0,
                motion_magnitude=0.0,
            ))
            start = end

        logger.info("Uniform segmentation: %d segments of ~%.1fs each", len(segments), window_sec)
        return segments

    def merge_short_segments(
        self,
        segments: List[ActionSegment],
        min_frames: int = 15,
    ) -> List[ActionSegment]:
        """
        Merge segments shorter than min_frames into their neighbours.

        Adjacent segments are merged if the shorter one is below threshold.
        This prevents single-frame "blip" segments.

        Parameters
        ----------
        segments : list of ActionSegment
        min_frames : int  minimum acceptable segment length

        Returns
        -------
        list of ActionSegment  (modified in place, returned for chaining)
        """
        if not segments:
            return segments

        merged = True
        while merged:
            merged = False
            new_segments: List[ActionSegment] = []
            i = 0
            while i < len(segments):
                seg = segments[i]
                if seg.num_frames < min_frames and i + 1 < len(segments):
                    # Merge with next segment
                    next_seg = segments[i + 1]
                    combined_mag = (
                        seg.motion_magnitude * seg.num_frames +
                        next_seg.motion_magnitude * next_seg.num_frames
                    ) / (seg.num_frames + next_seg.num_frames)
                    new_segments.append(ActionSegment(
                        start_frame=seg.start_frame,
                        end_frame=next_seg.end_frame,
                        confidence=max(seg.confidence, next_seg.confidence),
                        motion_magnitude=combined_mag,
                    ))
                    i += 2
                    merged = True
                elif seg.num_frames < min_frames and new_segments:
                    # Merge with previous segment
                    prev = new_segments[-1]
                    combined_mag = (
                        prev.motion_magnitude * prev.num_frames +
                        seg.motion_magnitude  * seg.num_frames
                    ) / (prev.num_frames + seg.num_frames)
                    new_segments[-1] = ActionSegment(
                        start_frame=prev.start_frame,
                        end_frame=seg.end_frame,
                        confidence=max(prev.confidence, seg.confidence),
                        motion_magnitude=combined_mag,
                    )
                    i += 1
                    merged = True
                else:
                    new_segments.append(seg)
                    i += 1
            segments = new_segments

        return segments

    # ── Internal helpers ─────────────────────────────────────────────────────────

    def _compute_flow_magnitudes(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute per-frame mean optical flow magnitude using Lucas-Kanade.

        Returns np.ndarray of shape (T-1,) where T = len(frames).
        """
        magnitudes = np.zeros(len(frames), dtype=np.float32)
        prev_gray  = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_pts   = cv2.goodFeaturesToTrack(prev_gray, mask=None, **_FEATURE_PARAMS)

        for i in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            if prev_pts is not None and len(prev_pts) > 0:
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, prev_pts, None, **_LK_PARAMS
                )
                valid = (status.ravel() == 1)
                if valid.sum() > 0:
                    delta = (curr_pts[valid] - prev_pts[valid]).reshape(-1, 2)
                    magnitudes[i] = float(np.mean(np.linalg.norm(delta, axis=1)))
                else:
                    magnitudes[i] = 0.0
            else:
                magnitudes[i] = 0.0

            # Refresh feature points periodically to avoid drift
            if i % 10 == 0 or (prev_pts is None or len(prev_pts) < 10):
                prev_pts = cv2.goodFeaturesToTrack(curr_gray, mask=None, **_FEATURE_PARAMS)
            else:
                if curr_pts is not None:
                    prev_pts = curr_pts[status.ravel() == 1].reshape(-1, 1, 2)

            prev_gray = curr_gray

        return magnitudes

    def _threshold_to_segments(
        self,
        magnitudes: np.ndarray,
    ) -> List[ActionSegment]:
        """
        Convert a time-series of motion magnitudes into ActionSegments.

        Uses a hysteresis threshold: segment starts when mag > high, ends when < low.
        """
        segments: List[ActionSegment] = []
        in_segment = False
        seg_start  = 0
        seg_mags: List[float] = []

        for i, mag in enumerate(magnitudes):
            if not in_segment:
                if mag > self.flow_threshold_high:
                    in_segment = True
                    seg_start  = i
                    seg_mags   = [mag]
            else:
                seg_mags.append(mag)
                if mag < self.flow_threshold_low:
                    mean_mag = float(np.mean(seg_mags))
                    # Confidence scales with how far above threshold the mean is
                    conf = min(1.0, mean_mag / (self.flow_threshold_high * 2))
                    segments.append(ActionSegment(
                        start_frame=seg_start,
                        end_frame=i,
                        confidence=conf,
                        motion_magnitude=mean_mag,
                    ))
                    in_segment = False
                    seg_mags   = []

        # Close any open segment
        if in_segment and seg_mags:
            mean_mag = float(np.mean(seg_mags))
            conf = min(1.0, mean_mag / (self.flow_threshold_high * 2))
            segments.append(ActionSegment(
                start_frame=seg_start,
                end_frame=len(magnitudes),
                confidence=conf,
                motion_magnitude=mean_mag,
            ))

        # If no segments found, return one covering all frames
        if not segments:
            overall_mean = float(np.mean(magnitudes))
            segments = [ActionSegment(0, len(magnitudes), 0.5, overall_mean)]

        return segments

"""
AutoLabeler: Generates automatic label suggestions before human review.

Uses:
  - Whisper transcription → language instruction + inferred action
  - Optical flow magnitude → motion intensity
  - GPS speed + place tags → scenario type
  - Simple keyword matching for Hindi/English action words

Output feeds into LabelStudio as pre-annotations, saving annotators time.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Hindi/English keyword → action_type mapping ───────────────────────────────

ACTION_KEYWORDS: Dict[str, List[str]] = {
    "pick_up":      ["उठाओ", "उठाना", "pick", "lift", "grab", "लेना", "ले"],
    "put_down":     ["रखो", "रखना", "place", "put", "drop", "रख"],
    "hand_over":    ["दो", "देना", "give", "hand", "deliver", "सौंपना", "दे"],
    "open_door":    ["खोलो", "खोलना", "open", "unlock", "खोल"],
    "close_door":   ["बंद करो", "बंद", "close", "lock", "बंद कर"],
    "navigate":     ["चलो", "जाओ", "go", "walk", "move", "जा", "चल", "आगे"],
    "driving":      ["drive", "ड्राइव", "गाड़ी", "bike", "चलाओ"],
    "scan_barcode": ["scan", "स्कैन", "barcode", "qr"],
    "pay_collect":  ["payment", "pay", "पैसे", "cash", "upi", "amount"],
    "idle":         ["wait", "रुको", "stop", "रुक", "खड़े"],
}

SCENARIO_KEYWORDS: Dict[str, List[str]] = {
    "food_delivery":  ["delivery", "food", "खाना", "order", "restaurant", "deliver"],
    "quick_commerce": ["grocery", "groceries", "kirana", "warehouse", "pick", "pack", "scan"],
    "driving":        ["drive", "road", "traffic", "signal", "turn", "lane"],
    "kitchen":        ["cook", "kitchen", "खाना बनाना", "gas", "stove", "pan", "chop"],
    "warehouse":      ["shelf", "rack", "box", "package", "sort", "inventory"],
}


@dataclass
class AutoLabel:
    """Predicted label for one action segment."""
    segment_index:   int
    action_type:     str                    # from ACTION_KEYWORDS keys
    scenario:        str                    # from SCENARIO_KEYWORDS keys
    instruction_hi:  str                    # Hindi instruction (from Whisper or inferred)
    instruction_en:  str                    # English instruction
    confidence:      float                  # 0–1
    motion_level:    str                    # "high" | "medium" | "low"
    is_stationary:   bool
    sources:         List[str] = field(default_factory=list)  # which signals contributed


class AutoLabeler:
    """
    Generates pre-annotation suggestions for action segments.

    Parameters
    ----------
    default_scenario : str
        Fallback scenario type if GPS/keyword matching fails.
    """

    def __init__(self, default_scenario: str = "unknown") -> None:
        self.default_scenario = default_scenario

    # ── Public ────────────────────────────────────────────────────────────────

    def label_segments(
        self,
        segments: List[Dict],
        frames: List[np.ndarray],
        transcription: Optional[Dict]  = None,
        gps_df=None,                         # pd.DataFrame from GPSPreprocessor
        imu_array: Optional[np.ndarray] = None,
    ) -> List[AutoLabel]:
        """
        Generate one AutoLabel per segment.

        Parameters
        ----------
        segments : list of {start_frame, end_frame, confidence}
        frames : list of (H, W, 3) uint8 arrays
        transcription : Whisper output {text, segments: [{text, start, end}]}
        gps_df : processed GPS DataFrame (optional)
        imu_array : (T, 6+) float32 IMU per-frame array (optional)

        Returns
        -------
        list of AutoLabel, one per segment.
        """
        trans_segs  = (transcription or {}).get("segments", [])
        full_text   = (transcription or {}).get("text", "")
        global_scenario = self._detect_scenario(full_text, gps_df)

        labels = []
        for seg_idx, seg in enumerate(segments):
            s, e = seg["start_frame"], seg["end_frame"]

            # 1. Get overlapping transcription
            seg_text = self._get_segment_text(s, e, trans_segs)

            # 2. Motion level from optical flow magnitude in segment
            motion_level, motion_conf = self._motion_level(frames, s, e)

            # 3. Stationarity from GPS
            is_stationary, gps_conf = self._is_stationary(gps_df, s, e)

            # 4. Action type from text + motion
            action_type, action_conf = self._detect_action(
                seg_text, motion_level, is_stationary
            )

            # 5. Scenario from text + GPS
            scenario = self._detect_scenario(seg_text, gps_df) or global_scenario

            # 6. Build instructions
            instr_hi, instr_en = self._build_instructions(
                seg_text, action_type, scenario
            )

            # 7. Overall confidence
            confidence = round(
                0.4 * action_conf + 0.3 * motion_conf + 0.3 * gps_conf, 2
            )

            sources = ["motion"]
            if seg_text:
                sources.append("whisper")
            if gps_df is not None:
                sources.append("gps")

            labels.append(AutoLabel(
                segment_index=seg_idx,
                action_type=action_type,
                scenario=scenario,
                instruction_hi=instr_hi,
                instruction_en=instr_en,
                confidence=confidence,
                motion_level=motion_level,
                is_stationary=is_stationary,
                sources=sources,
            ))

        return labels

    def to_labelstudio_format(self, labels: List[AutoLabel]) -> List[Dict]:
        """Convert AutoLabel list to dicts for LabelStudioClient.push_segment_tasks."""
        return [
            {
                "action_type":    lb.action_type,
                "scenario":       lb.scenario,
                "instruction_hi": lb.instruction_hi,
                "instruction_en": lb.instruction_en,
                "confidence":     lb.confidence,
            }
            for lb in labels
        ]

    def summary(self, labels: List[AutoLabel]) -> Dict[str, Any]:
        """Return aggregate stats about auto-labels."""
        from collections import Counter
        return {
            "total":            len(labels),
            "action_types":     dict(Counter(lb.action_type for lb in labels)),
            "scenarios":        dict(Counter(lb.scenario    for lb in labels)),
            "motion_levels":    dict(Counter(lb.motion_level for lb in labels)),
            "avg_confidence":   round(np.mean([lb.confidence for lb in labels]), 3) if labels else 0.0,
            "stationary_count": sum(lb.is_stationary for lb in labels),
        }

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _detect_action(
        self,
        text: str,
        motion_level: str,
        is_stationary: bool,
    ) -> Tuple[str, float]:
        """Return (action_type, confidence)."""
        text_lower = text.lower()

        # Text-based match
        for action, keywords in ACTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return action, 0.85

        # Motion-based fallback
        if is_stationary:
            return "idle", 0.6
        if motion_level == "high":
            return "navigate", 0.5
        if motion_level == "medium":
            return "navigate", 0.4
        return "other", 0.3

    def _detect_scenario(self, text: str, gps_df=None) -> str:
        text_lower = (text or "").lower()

        # Text match
        for scenario, keywords in SCENARIO_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return scenario

        # GPS speed-based: if average speed > 15 km/h → driving
        if gps_df is not None and "speed_kmh" in gps_df.columns:
            avg_speed = float(gps_df["speed_kmh"].mean())
            if avg_speed > 15:
                return "driving"
            if avg_speed > 3:
                return "food_delivery"  # likely rider

        return self.default_scenario

    def _motion_level(
        self, frames: List[np.ndarray], start: int, end: int
    ) -> Tuple[str, float]:
        """Compute optical flow magnitude for a segment → motion level."""
        if not frames or end <= start + 1:
            return "low", 0.4

        try:
            import cv2
            seg_frames = frames[start: min(end, start + 10)]  # sample up to 10
            magnitudes = []
            for i in range(len(seg_frames) - 1):
                g1 = cv2.cvtColor(seg_frames[i],   cv2.COLOR_BGR2GRAY) if seg_frames[i].ndim == 3 else seg_frames[i]
                g2 = cv2.cvtColor(seg_frames[i+1], cv2.COLOR_BGR2GRAY) if seg_frames[i+1].ndim == 3 else seg_frames[i+1]
                flow = cv2.calcOpticalFlowFarneback(
                    g1.astype(np.uint8), g2.astype(np.uint8),
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                magnitudes.append(float(mag))

            avg_mag = np.mean(magnitudes) if magnitudes else 0.0
            if avg_mag > 8.0:
                return "high",   0.8
            if avg_mag > 2.0:
                return "medium", 0.7
            return "low", 0.6
        except Exception:
            return "low", 0.3

    def _is_stationary(
        self, gps_df, start_frame: int, end_frame: int
    ) -> Tuple[bool, float]:
        """Check if GPS says this segment is stationary."""
        if gps_df is None or "is_stationary" not in gps_df.columns:
            return False, 0.3

        # Map frame range to GPS rows (GPS is 30fps after interpolation)
        seg = gps_df.iloc[start_frame: end_frame]
        if len(seg) == 0:
            return False, 0.3

        frac_stationary = seg["is_stationary"].mean()
        is_stat = frac_stationary > 0.6
        return is_stat, 0.75

    def _get_segment_text(
        self, start_frame: int, end_frame: int, trans_segs: List[Dict]
    ) -> str:
        """Collect all Whisper transcript text that overlaps this segment."""
        start_sec = start_frame / 30.0
        end_sec   = end_frame   / 30.0
        texts: List[str] = []
        for ts in trans_segs:
            t_start = float(ts.get("start", 0))
            t_end   = float(ts.get("end",   0))
            overlap = max(0.0, min(end_sec, t_end) - max(start_sec, t_start))
            if overlap > 0.1:
                texts.append(str(ts.get("text", "")).strip())
        return " ".join(texts).strip()

    def _build_instructions(
        self, text: str, action_type: str, scenario: str
    ) -> Tuple[str, str]:
        """Build Hindi and English instruction strings."""
        # If we have transcription text, use it directly
        if text and len(text) > 3:
            hi = text
            en = self._translate_hint(text, action_type)
            return hi, en

        # Template-based fallback
        ACTION_TEMPLATES_HI: Dict[str, str] = {
            "pick_up":      "वस्तु उठाओ",
            "put_down":     "वस्तु रखो",
            "hand_over":    "सामान सौंपो",
            "open_door":    "दरवाजा खोलो",
            "close_door":   "दरवाजा बंद करो",
            "navigate":     "आगे बढ़ो",
            "driving":      "गाड़ी चलाओ",
            "scan_barcode": "बारकोड स्कैन करो",
            "pay_collect":  "पेमेंट लो",
            "idle":         "प्रतीक्षा करो",
        }
        ACTION_TEMPLATES_EN: Dict[str, str] = {
            "pick_up":      "Pick up the item",
            "put_down":     "Put down the item",
            "hand_over":    "Hand over the package",
            "open_door":    "Open the door",
            "close_door":   "Close the door",
            "navigate":     "Navigate forward",
            "driving":      "Drive the vehicle",
            "scan_barcode": "Scan the barcode",
            "pay_collect":  "Collect payment",
            "idle":         "Wait",
        }
        hi = ACTION_TEMPLATES_HI.get(action_type, "कार्य करो")
        en = ACTION_TEMPLATES_EN.get(action_type, "Perform task")
        return hi, en

    def _translate_hint(self, text: str, action_type: str) -> str:
        """Very rough Hindi→English hint (keyword swap, not a real translator)."""
        replacements = {
            "उठाओ": "pick up", "रखो": "put down", "दो": "give",
            "खोलो": "open", "बंद": "close", "चलो": "go",
            "स्कैन": "scan", "पैसे": "payment", "रुको": "wait",
        }
        result = text
        for hi, en in replacements.items():
            result = result.replace(hi, en)
        return result if result != text else f"[{action_type}]"

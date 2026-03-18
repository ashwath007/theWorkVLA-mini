"""
HindiTranscriber: Speech-to-text for Hindi and English using OpenAI Whisper.

Handles audio arrays or file paths, returns structured transcript segments
with start/end times and detected language.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single transcript segment from Whisper output."""
    text: str
    start: float        # seconds
    end: float          # seconds
    language: str       # ISO 639-1 code, e.g., 'hi', 'en'
    confidence: float   # average token log-prob converted to [0,1]

    def to_dict(self) -> dict:
        return {
            "text":       self.text,
            "start":      round(self.start, 3),
            "end":        round(self.end, 3),
            "language":   self.language,
            "confidence": round(self.confidence, 4),
        }


class HindiTranscriber:
    """
    Speech-to-text transcriber targeting Hindi + English code-mixing.

    Uses OpenAI Whisper for transcription.  The model is loaded lazily on first
    call to avoid startup overhead when not needed.

    Parameters
    ----------
    model_name : str
        Whisper model size: 'tiny', 'base', 'small', 'medium', 'large'.
        'tiny' is the default for low-resource (Raspberry Pi) environments.
    device : str
        Torch device for inference ('cpu', 'cuda', 'mps').
    default_language : str
        Default language hint ('hi' for Hindi, None for auto-detect).
    """

    def __init__(
        self,
        model_name: str = "tiny",
        device: Optional[str] = None,
        default_language: str = "hi",
    ) -> None:
        self.model_name       = model_name
        self.default_language = default_language
        self._model           = None

        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _ensure_model(self) -> Any:
        """Load Whisper model lazily."""
        if self._model is None:
            try:
                import whisper
                logger.info("Loading Whisper model '%s' on %s …", self.model_name, self.device)
                self._model = whisper.load_model(self.model_name, device=self.device)
                logger.info("Whisper model loaded.")
            except ImportError as exc:
                raise ImportError(
                    "openai-whisper is required for transcription. "
                    "Install with: pip install openai-whisper"
                ) from exc
        return self._model

    # ── Public API ──────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path_or_array: Union[str, np.ndarray],
        language: Optional[str] = None,
        sr: int = 16000,
    ) -> List[TranscriptSegment]:
        """
        Transcribe audio and return a list of timed segments.

        Parameters
        ----------
        audio_path_or_array : str or np.ndarray
            Audio file path OR float32 numpy array (mono, 16 kHz).
        language : str, optional
            ISO 639-1 language code ('hi', 'en', etc.).
            If None, Whisper auto-detects.
        sr : int
            Sample rate if audio_path_or_array is an ndarray.

        Returns
        -------
        list of TranscriptSegment
        """
        model = self._ensure_model()
        lang  = language or self.default_language

        if isinstance(audio_path_or_array, np.ndarray):
            audio_input = self._array_to_tmp_file(audio_path_or_array, sr)
            cleanup     = True
        else:
            audio_input = audio_path_or_array
            cleanup     = False

        try:
            options: Dict[str, Any] = {"language": lang} if lang else {}
            result = model.transcribe(audio_input, **options)
        finally:
            if cleanup and os.path.exists(audio_input):
                os.unlink(audio_input)

        detected_lang = result.get("language", lang or "?")
        segments      = result.get("segments", [])

        return [
            TranscriptSegment(
                text=seg["text"].strip(),
                start=float(seg["start"]),
                end=float(seg["end"]),
                language=detected_lang,
                confidence=self._avg_confidence(seg),
            )
            for seg in segments
            if seg["text"].strip()
        ]

    def transcribe_chunk(
        self,
        audio_array: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        """
        Transcribe a single short audio chunk (e.g., one action segment).

        The chunk is padded or trimmed to a minimum of 1 second to avoid
        Whisper decoding errors on very short inputs.

        Parameters
        ----------
        audio_array : np.ndarray  float32  mono
        sr : int  sample rate (should be 16000)
        language : str, optional

        Returns
        -------
        list of TranscriptSegment (usually 1 element for short clips)
        """
        min_samples = sr  # 1 second minimum
        if len(audio_array) < min_samples:
            padded = np.zeros(min_samples, dtype=np.float32)
            padded[: len(audio_array)] = audio_array
            audio_array = padded

        return self.transcribe(audio_array, language=language, sr=sr)

    def transcribe_segments(
        self,
        audio_array: np.ndarray,
        segment_times: List[Dict[str, float]],
        sr: int = 16000,
        language: Optional[str] = None,
    ) -> List[Dict]:
        """
        Transcribe audio for each segment defined by start/end times.

        Parameters
        ----------
        audio_array : np.ndarray  float32 mono full audio
        segment_times : list of dicts with 'start' and 'end' keys (seconds)
        sr : int  sample rate
        language : str, optional

        Returns
        -------
        list of dicts: same as segment_times input, but with 'text', 'language',
        'confidence' keys added.
        """
        results = []
        for seg in segment_times:
            start_sample = int(seg["start"] * sr)
            end_sample   = int(seg["end"]   * sr)
            chunk        = audio_array[start_sample:end_sample]

            transcripts = self.transcribe_chunk(chunk, sr=sr, language=language)
            text  = " ".join(t.text for t in transcripts) if transcripts else ""
            conf  = np.mean([t.confidence for t in transcripts]) if transcripts else 0.0
            lang  = transcripts[0].language if transcripts else (language or "?")

            results.append({
                **seg,
                "text":       text,
                "language":   lang,
                "confidence": float(conf),
            })
        return results

    # ── Internal helpers ─────────────────────────────────────────────────────────

    def _array_to_tmp_file(self, audio: np.ndarray, sr: int) -> str:
        """Save numpy float32 audio to a temporary WAV file, return path."""
        import wave
        import struct

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        # Convert float32 → int16
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_int = (pcm * 32767).astype(np.int16)

        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_int.tobytes())

        return tmp.name

    @staticmethod
    def _avg_confidence(segment: dict) -> float:
        """Compute average confidence from Whisper token log-probs."""
        tokens = segment.get("tokens", [])
        if not tokens:
            return 0.5
        # Whisper avg_logprob is in [-inf, 0]; convert to [0, 1] via exp
        avg_logprob = segment.get("avg_logprob", -0.5)
        return float(np.clip(np.exp(avg_logprob), 0.0, 1.0))

"""
PipelineRunner: End-to-end orchestrator for the VLA data pipeline.

Chains all preprocessing stages in order:
    assemble → sync → video → audio → imu → gps → transcribe → segment → chunk → store

Each stage writes a checkpoint file so the pipeline can resume after a crash.
Run from CLI:
    python -m src.pipeline.runner --session-dir /data/sessions/my-session
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(name="pipeline", help="VLA data pipeline runner")


# ── Stage definitions ─────────────────────────────────────────────────────────

class Stage(str, Enum):
    ASSEMBLE   = "assemble"
    SYNC       = "sync"
    VIDEO      = "video"
    AUDIO      = "audio"
    IMU        = "imu"
    GPS        = "gps"
    TRANSCRIBE = "transcribe"
    SEGMENT    = "segment"
    CHUNK      = "chunk"
    VALIDATE   = "validate"
    STORE      = "store"
    DONE       = "done"


STAGE_ORDER: List[Stage] = [
    Stage.ASSEMBLE,
    Stage.SYNC,
    Stage.VIDEO,
    Stage.AUDIO,
    Stage.IMU,
    Stage.GPS,
    Stage.TRANSCRIBE,
    Stage.SEGMENT,
    Stage.CHUNK,
    Stage.VALIDATE,
    Stage.STORE,
    Stage.DONE,
]


@dataclass
class StageResult:
    stage: str
    status: str          # ok | skipped | error
    duration_sec: float
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineState:
    session_id: str
    session_dir: str
    current_stage: str
    completed_stages: List[str]
    results: Dict[str, Any]          # stage → StageResult dict
    started_at: str
    updated_at: str
    is_done: bool = False
    error: Optional[str] = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        data = json.loads(path.read_text())
        return cls(**data)

    @classmethod
    def new(cls, session_id: str, session_dir: str) -> "PipelineState":
        now = datetime.utcnow().isoformat()
        return cls(
            session_id=session_id,
            session_dir=session_dir,
            current_stage=Stage.ASSEMBLE,
            completed_stages=[],
            results={},
            started_at=now,
            updated_at=now,
        )


# ── PipelineRunner ────────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Runs every preprocessing stage for one session directory.

    Parameters
    ----------
    session_dir : str | Path
        Root directory of the session (contains chunks/ or assembled/).
    resume : bool
        If True, skip stages already marked complete in the checkpoint.
    skip_stages : list[str]
        Stage names to skip (e.g. ["store"] when testing offline).
    upload_to_hf : bool
        Whether to push episodes to HuggingFace after chunking.
    hf_repo : str
        HuggingFace dataset repo id.
    """

    def __init__(
        self,
        session_dir: str | Path,
        resume: bool = True,
        skip_stages: Optional[List[str]] = None,
        upload_to_hf: bool = False,
        hf_repo: str = "",
        scenario_type: str = "unknown",
    ) -> None:
        self.session_dir  = Path(session_dir)
        self.resume       = resume
        self.skip_stages  = set(skip_stages or [])
        self.upload_to_hf = upload_to_hf
        self.hf_repo      = hf_repo
        self.scenario_type = scenario_type

        self.session_id   = self.session_dir.name
        self.checkpoint   = self.session_dir / ".pipeline_state.json"
        self.assembled_dir = self.session_dir / "assembled"
        self.episodes_dir  = self.session_dir / "episodes"

        # Shared state passed between stages
        self._ctx: Dict[str, Any] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> PipelineState:
        """Execute all stages in order. Returns final PipelineState."""
        state = self._load_or_create_state()
        console.rule(f"[bold cyan]Pipeline: {self.session_id}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            for stage in STAGE_ORDER:
                if stage == Stage.DONE:
                    break

                if self.resume and stage.value in state.completed_stages:
                    console.print(f"  [dim]↩ {stage.value} (already done)")
                    continue

                if stage.value in self.skip_stages:
                    console.print(f"  [yellow]⊘ {stage.value} (skipped)")
                    state.results[stage.value] = asdict(
                        StageResult(stage.value, "skipped", 0.0)
                    )
                    continue

                task = progress.add_task(f"[cyan]{stage.value}...", total=None)
                result = self._run_stage(stage, state)
                progress.remove_task(task)

                state.results[stage.value] = asdict(result)
                if result.status == "ok":
                    state.completed_stages.append(stage.value)
                    console.print(f"  [green]✓ {stage.value}[/green]  ({result.duration_sec:.1f}s)")
                elif result.status == "skipped":
                    console.print(f"  [yellow]⊘ {stage.value}[/yellow]  (module unavailable)")
                else:
                    console.print(f"  [red]✗ {stage.value}[/red]  {result.error}")
                    state.error = result.error
                    state.updated_at = datetime.utcnow().isoformat()
                    state.save(self.checkpoint)
                    # Continue pipeline even on non-fatal errors
                    if stage in (Stage.ASSEMBLE, Stage.SYNC):
                        console.print("  [red]Fatal stage failed — stopping pipeline.")
                        return state

                state.current_stage = stage.value
                state.updated_at    = datetime.utcnow().isoformat()
                state.save(self.checkpoint)

        state.is_done      = True
        state.current_stage = Stage.DONE
        state.updated_at   = datetime.utcnow().isoformat()
        state.save(self.checkpoint)

        self._print_summary(state)
        return state

    def status(self) -> Optional[PipelineState]:
        """Return current pipeline state without running."""
        if self.checkpoint.exists():
            return PipelineState.load(self.checkpoint)
        return None

    # ── Stage dispatch ────────────────────────────────────────────────────────

    def _run_stage(self, stage: Stage, state: PipelineState) -> StageResult:
        handler: Dict[Stage, Callable] = {
            Stage.ASSEMBLE:   self._stage_assemble,
            Stage.SYNC:       self._stage_sync,
            Stage.VIDEO:      self._stage_video,
            Stage.AUDIO:      self._stage_audio,
            Stage.IMU:        self._stage_imu,
            Stage.GPS:        self._stage_gps,
            Stage.TRANSCRIBE: self._stage_transcribe,
            Stage.SEGMENT:    self._stage_segment,
            Stage.CHUNK:      self._stage_chunk,
            Stage.VALIDATE:   self._stage_validate,
            Stage.STORE:      self._stage_store,
        }
        fn = handler.get(stage)
        if fn is None:
            return StageResult(stage.value, "skipped", 0.0)

        t0 = time.perf_counter()
        try:
            output = fn()
            duration = time.perf_counter() - t0
            return StageResult(stage.value, "ok", round(duration, 2), output=output or {})
        except Exception as exc:
            duration = time.perf_counter() - t0
            logger.error("Stage %s failed: %s", stage.value, exc, exc_info=True)
            return StageResult(stage.value, "error", round(duration, 2), error=str(exc))

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage_assemble(self) -> Dict:
        from ..preprocess.assembler import ChunkAssembler
        assembler = ChunkAssembler()
        session = assembler.assemble_session(str(self.session_dir))
        self._ctx["assembled"] = session
        self.assembled_dir.mkdir(parents=True, exist_ok=True)
        return {
            "video_path":        session.video_path,
            "audio_path":        session.audio_path,
            "imu_path":          session.imu_path,
            "gps_path":          session.gps_path,
            "total_duration_sec": session.total_duration_sec,
            "chunk_count":       session.chunk_count,
        }

    def _stage_sync(self) -> Dict:
        from ..preprocess.sync import StreamSynchronizer
        assembled = self._ctx.get("assembled")
        if assembled is None:
            # Try to find assembled files directly
            video = str(self.assembled_dir / "video.mp4")
            audio = str(self.assembled_dir / "audio.wav")
            imu   = str(self.assembled_dir / "imu.csv")
            meta  = str(self.assembled_dir / "session_meta.json")
        else:
            video = assembled.video_path
            audio = assembled.audio_path
            imu   = assembled.imu_path
            meta  = str(self.assembled_dir / "session_meta.json")

        syncer = StreamSynchronizer()
        synced = syncer.align_streams(
            video_path=video,
            audio_path=audio,
            imu_csv_path=imu,
            metadata_path=meta,
        )
        self._ctx["synced"] = synced
        return {"num_frames": synced["num_frames"], "fps": synced["fps"]}

    def _stage_video(self) -> Dict:
        from ..preprocess.video import VideoPreprocessor
        synced = self._ctx.get("synced", {})
        video_path = synced.get("video_path") or str(self.assembled_dir / "video.mp4")

        proc = VideoPreprocessor()
        frames = proc.extract_frames(video_path)
        frames = proc.blur_faces(frames)
        frames_resized = proc.resize_frames(frames)
        frames_norm    = proc.normalize_frames(frames_resized)

        self._ctx["frames"]      = frames_resized   # uint8 for storage
        self._ctx["frames_norm"] = frames_norm       # float32 for model
        return {"num_frames": len(frames), "shape": list(frames_resized[0].shape) if frames_resized else []}

    def _stage_audio(self) -> Dict:
        from ..preprocess.audio import AudioPreprocessor
        assembled = self._ctx.get("assembled")
        audio_path = assembled.audio_path if assembled else str(self.assembled_dir / "audio.wav")

        proc = AudioPreprocessor()
        samples, sr = proc.load_audio(audio_path)
        samples = proc.resample_to_16k(samples, sr)
        samples = proc.normalize_audio(samples)
        self._ctx["audio_samples"] = samples
        self._ctx["audio_sr"]      = 16000
        return {"duration_sec": round(len(samples) / 16000, 1), "sample_rate": 16000}

    def _stage_imu(self) -> Dict:
        from ..preprocess.imu import IMUPreprocessor
        assembled = self._ctx.get("assembled")
        imu_path = assembled.imu_path if assembled else str(self.assembled_dir / "imu.csv")

        proc = IMUPreprocessor()
        df = proc.load_imu(imu_path)
        df = proc.remove_gravity(df)
        df = proc.normalize_imu(df)
        df = proc.interpolate_to_fps(df, target_fps=30)
        self._ctx["imu_df"] = df
        return {"num_readings": len(df), "columns": list(df.columns)}

    def _stage_gps(self) -> Dict:
        from ..preprocess.gps import GPSPreprocessor
        assembled = self._ctx.get("assembled")
        gps_path = assembled.gps_path if assembled else str(self.assembled_dir / "gps.csv")

        if not Path(gps_path).exists():
            logger.warning("GPS file not found at %s — skipping GPS stage", gps_path)
            self._ctx["gps_df"] = None
            return {"status": "no_gps_file"}

        proc = GPSPreprocessor()
        df = proc.load_gps(gps_path)
        df = proc.kalman_smooth(df)
        df = proc.compute_derived(df)
        df = proc.tag_places(df, nominatim_offline=True)
        df = proc.interpolate_to_fps(df, fps=30)

        out_path = str(self.assembled_dir / "gps_processed.parquet")
        proc.save_processed(df, out_path)
        self._ctx["gps_df"] = df

        # Extract place summary for metadata
        places = df["place_name"].value_counts().head(3).to_dict() if "place_name" in df.columns else {}
        total_dist = df["distance_m"].max() if "distance_m" in df.columns else 0
        return {
            "num_points":  len(df),
            "total_dist_m": round(float(total_dist), 1),
            "top_places":  {str(k): int(v) for k, v in places.items()},
        }

    def _stage_transcribe(self) -> Dict:
        from ..segmentation.transcriber import HindiTranscriber
        assembled = self._ctx.get("assembled")
        audio_path = assembled.audio_path if assembled else str(self.assembled_dir / "audio.wav")

        transcriber = HindiTranscriber()
        result = transcriber.transcribe(audio_path, language="hi")
        self._ctx["transcription"] = result

        segments = result.get("segments", [])
        text     = result.get("text", "")
        out_path = self.assembled_dir / "transcription.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return {
            "text_preview":   text[:120],
            "num_segments":   len(segments),
            "language":       result.get("language", "hi"),
        }

    def _stage_segment(self) -> Dict:
        from ..segmentation.action_segmenter import ActionSegmenter
        frames = self._ctx.get("frames", [])
        if not frames:
            raise RuntimeError("No frames in context — video stage must run first")

        segmenter = ActionSegmenter()
        segments  = segmenter.segment_optical_flow(frames)
        if not segments:
            segments = segmenter.segment_uniform(frames)
        segments = segmenter.merge_short_segments(segments)
        self._ctx["segments"] = segments

        out_path = self.assembled_dir / "segments.json"
        out_path.write_text(json.dumps(
            [{"start": s.start_frame, "end": s.end_frame, "confidence": s.confidence}
             for s in segments],
            indent=2,
        ))
        return {"num_segments": len(segments)}

    def _stage_chunk(self) -> Dict:
        from ..segmentation.lerobot_chunker import LeRobotChunker
        from ..preprocess.hdf5_writer import HDF5Writer

        synced      = self._ctx.get("synced", {})
        frames      = self._ctx.get("frames", [])
        audio       = self._ctx.get("audio_samples")
        imu_df      = self._ctx.get("imu_df")
        gps_df      = self._ctx.get("gps_df")
        segments    = self._ctx.get("segments", [])
        transcription = self._ctx.get("transcription", {})

        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        writer  = HDF5Writer()
        chunker = LeRobotChunker()

        # Build per-frame imu array
        import numpy as np
        import pandas as pd
        T = len(frames) if frames else synced.get("num_frames", 0)

        imu_array = np.zeros((T, 10), dtype=np.float32)
        if imu_df is not None and len(imu_df) >= T:
            cols = [c for c in ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"] if c in imu_df.columns]
            for j, col in enumerate(cols[:6]):
                imu_array[:T, j] = imu_df[col].values[:T]

        gps_array = np.zeros((T, 6), dtype=np.float32)
        if gps_df is not None:
            from ..preprocess.gps import GPSPreprocessor
            gps_proc  = GPSPreprocessor()
            gps_embed = gps_proc.to_embedding_array(gps_df)
            n = min(T, len(gps_embed))
            gps_array[:n] = gps_embed[:n]

        # Audio chunks per frame
        sr = self._ctx.get("audio_sr", 16000)
        samples_per_frame = max(1, sr // 30)
        audio_array = np.zeros((T, samples_per_frame), dtype=np.float32)
        if audio is not None:
            for i in range(T):
                start = i * samples_per_frame
                end   = start + samples_per_frame
                chunk = audio[start:end]
                n = min(len(chunk), samples_per_frame)
                audio_array[i, :n] = chunk[:n]

        # Action: head-pose delta (finite diff of IMU quaternion columns)
        action_array = np.zeros((T, 7), dtype=np.float32)
        if imu_df is not None:
            action_array[:T, :6] = imu_array[:T, :6]

        # Language instruction from transcription
        lang = transcription.get("text", "record egocentric task")

        # Write episodes per segment
        episode_paths: List[str] = []
        trans_segs = transcription.get("segments", [])

        for ep_idx, seg in enumerate(segments):
            s, e = seg.start_frame, min(seg.end_frame, T)
            if e <= s:
                continue

            seg_lang = self._find_language_for_segment(seg, trans_segs) or lang

            ep_frames = np.array(frames[s:e]) if frames else np.zeros((e-s, 224, 224, 3), dtype=np.uint8)
            ep_path   = str(self.episodes_dir / f"episode_{ep_idx:06d}.h5")

            writer.write_session(
                session_id=f"{self.session_id}_ep{ep_idx:04d}",
                frames=ep_frames,
                audio_chunks=audio_array[s:e],
                imu_data=imu_array[s:e],
                metadata={
                    "session_id":    self.session_id,
                    "episode_index": ep_idx,
                    "scenario_type": self.scenario_type,
                    "fps":           30,
                    "language":      "hi",
                    "gps_embedding": gps_array[s:e].tolist() if gps_df is not None else [],
                },
                output_path=ep_path,
                language_instruction=seg_lang,
                actions=action_array[s:e],
            )
            episode_paths.append(ep_path)

        self._ctx["episode_paths"] = episode_paths

        # Write dataset_info.json
        dataset_info = {
            "session_id":    self.session_id,
            "num_episodes":  len(episode_paths),
            "scenario_type": self.scenario_type,
            "total_frames":  T,
            "fps":           30,
            "image_size":    [224, 224],
            "action_dim":    7,
            "created_at":    datetime.utcnow().isoformat(),
        }
        (self.episodes_dir / "dataset_info.json").write_text(
            json.dumps(dataset_info, indent=2)
        )
        return {"num_episodes": len(episode_paths), "episodes_dir": str(self.episodes_dir)}

    def _stage_validate(self) -> Dict:
        from .validator import DataValidator
        episode_paths = self._ctx.get("episode_paths", [])
        if not episode_paths:
            ep_dir = self.episodes_dir
            episode_paths = sorted(str(p) for p in ep_dir.glob("episode_*.h5"))

        validator = DataValidator()
        report    = validator.validate_episodes(episode_paths)
        out_path  = self.assembled_dir / "validation_report.json"
        out_path.write_text(json.dumps(report, indent=2))

        if report["invalid_count"] > 0:
            logger.warning(
                "%d/%d episodes failed validation",
                report["invalid_count"], report["total_count"],
            )
        return report

    def _stage_store(self) -> Dict:
        from ..storage.hdf5_store import HDF5Store
        episode_paths = self._ctx.get("episode_paths", [])
        if not episode_paths:
            episode_paths = sorted(
                str(p) for p in self.episodes_dir.glob("episode_*.h5")
            )

        store = HDF5Store(str(self.episodes_dir))
        stats = store.get_session_stats(str(self.episodes_dir))

        if self.upload_to_hf and self.hf_repo:
            from ..storage.hf_uploader import HFUploader
            import os
            token = os.environ.get("HF_TOKEN", "")
            if token:
                uploader = HFUploader(self.hf_repo, token)
                uploader.upload_dataset(str(self.episodes_dir), self.hf_repo)
                return {**stats, "uploaded_to_hf": True, "hf_repo": self.hf_repo}

        return {**stats, "uploaded_to_hf": False}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_language_for_segment(self, seg: Any, trans_segs: List[Dict]) -> Optional[str]:
        """Return transcription text that overlaps most with a segment."""
        if not trans_segs:
            return None
        seg_start_sec = seg.start_frame / 30.0
        seg_end_sec   = seg.end_frame   / 30.0
        best_text = ""
        best_overlap = 0.0
        for ts in trans_segs:
            t_start = float(ts.get("start", 0))
            t_end   = float(ts.get("end",   0))
            overlap = max(0.0, min(seg_end_sec, t_end) - max(seg_start_sec, t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_text    = str(ts.get("text", ""))
        return best_text.strip() or None

    def _load_or_create_state(self) -> PipelineState:
        if self.resume and self.checkpoint.exists():
            try:
                state = PipelineState.load(self.checkpoint)
                console.print(f"[dim]Resuming from checkpoint: {len(state.completed_stages)} stages done")
                return state
            except Exception as exc:
                logger.warning("Could not load checkpoint: %s — starting fresh", exc)
        return PipelineState.new(self.session_id, str(self.session_dir))

    def _print_summary(self, state: PipelineState) -> None:
        table = Table(title="Pipeline Summary", show_header=True)
        table.add_column("Stage",    style="cyan")
        table.add_column("Status",   style="bold")
        table.add_column("Duration", justify="right")
        table.add_column("Notes")

        for stage in STAGE_ORDER:
            if stage == Stage.DONE:
                continue
            r = state.results.get(stage.value)
            if r is None:
                table.add_row(stage.value, "[dim]not run", "-", "")
                continue
            status = r.get("status", "?")
            color  = {"ok": "green", "skipped": "yellow", "error": "red"}.get(status, "white")
            dur    = f"{r.get('duration_sec', 0):.1f}s"
            notes  = r.get("error") or ""
            if not notes:
                output = r.get("output", {})
                notes  = "  ".join(f"{k}={v}" for k, v in list(output.items())[:2])
            table.add_row(stage.value, f"[{color}]{status}", dur, str(notes)[:60])

        console.print(table)


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command("run")
def cli_run(
    session_dir:  str  = typer.Argument(..., help="Path to session directory"),
    resume:       bool = typer.Option(True,  "--resume/--no-resume", help="Resume from checkpoint"),
    skip:         str  = typer.Option("",    "--skip", help="Comma-separated stages to skip"),
    upload_hf:    bool = typer.Option(False, "--upload-hf", help="Upload to HuggingFace after chunking"),
    hf_repo:      str  = typer.Option("",    "--hf-repo", help="HuggingFace repo id"),
    scenario:     str  = typer.Option("unknown", "--scenario", help="delivery|driving|warehouse|kitchen"),
) -> None:
    """Run the full VLA pipeline on a session directory."""
    skip_list = [s.strip() for s in skip.split(",") if s.strip()]
    runner = PipelineRunner(
        session_dir=session_dir,
        resume=resume,
        skip_stages=skip_list,
        upload_to_hf=upload_hf,
        hf_repo=hf_repo,
        scenario_type=scenario,
    )
    state = runner.run()
    if state.error:
        raise typer.Exit(code=1)


@app.command("status")
def cli_status(
    session_dir: str = typer.Argument(..., help="Path to session directory"),
) -> None:
    """Show pipeline status for a session."""
    runner = PipelineRunner(session_dir=session_dir)
    state  = runner.status()
    if state is None:
        console.print("[yellow]No pipeline checkpoint found.")
        return
    console.print(f"Session:  {state.session_id}")
    console.print(f"Stage:    {state.current_stage}")
    console.print(f"Done:     {state.is_done}")
    console.print(f"Stages completed: {', '.join(state.completed_stages)}")


if __name__ == "__main__":
    app()

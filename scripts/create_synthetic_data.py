"""
create_synthetic_data.py — Generate synthetic training data for offline testing.

Creates fake sessions with random frames, audio, IMU, and Hindi instruction strings,
saved in LeRobot-compatible HDF5 format.

Usage
-----
    python scripts/create_synthetic_data.py \\
        --num-sessions 10 \\
        --frames-per-session 100 \\
        --output-dir /data/synthetic_episodes
"""

from __future__ import annotations

import csv
import json
import os
import sys
import uuid
import wave
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, SpinnerColumn, TextColumn, TimeRemainingColumn

# Add project src to path
_SCRIPT_DIR  = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

app     = typer.Typer(name="create-synthetic-data", add_completion=False)
console = Console()

# ── Hindi instruction templates ───────────────────────────────────────────────

_HINDI_INSTRUCTIONS = [
    "बॉक्स उठाओ और उसे दूसरी जगह रखो",        # Pick up the box and place it elsewhere
    "दरवाज़ा खोलो",                              # Open the door
    "पानी की बोतल उठाओ",                         # Pick up the water bottle
    "मेज़ पर रखी चीज़ें व्यवस्थित करो",          # Organize the things on the table
    "लाल डिब्बे को शेल्फ पर रखो",              # Put the red box on the shelf
    "फ़ाइल को अलमारी में रखो",                    # Put the file in the cupboard
    "बटन दबाओ",                                   # Press the button
    "चाय बनाओ",                                   # Make tea
    "सब्ज़ियाँ काटो",                             # Cut the vegetables
    "कमरा साफ़ करो",                              # Clean the room
    "लैपटॉप चालू करो",                           # Turn on the laptop
    "खिड़की बंद करो",                             # Close the window
    "पुस्तक शेल्फ पर रखो",                       # Place the book on the shelf
    "मशीन के पुर्जे जोड़ो",                      # Assemble machine parts
    "कागज़ को फ़ोल्ड करो और लिफ़ाफ़े में रखो",   # Fold the paper and put in envelope
    "Open the box and check contents",           # English for code-mixing
    "Pick up the tool from the table",
    "Assemble the component",
    "Check the circuit board",
    "Stack the boxes neatly",
]


def _random_instruction() -> str:
    return np.random.choice(_HINDI_INSTRUCTIONS)


# ── Data generators ───────────────────────────────────────────────────────────

def _generate_frames(n: int, h: int = 224, w: int = 224) -> np.ndarray:
    """Generate random uint8 RGB frames with some structure."""
    rng    = np.random.default_rng()
    frames = rng.integers(50, 200, size=(n, h, w, 3), dtype=np.uint8)
    # Add a fake "horizon line" for visual realism
    horizon = h // 2
    frames[:, :horizon, :, :] = frames[:, :horizon, :, :] // 2 + 100  # lighter top
    return frames


def _generate_audio_chunks(n: int, sr: int = 16000, fps: int = 30) -> np.ndarray:
    """Generate random float32 audio (one chunk per frame)."""
    samples_per_frame = sr // fps
    rng    = np.random.default_rng()
    chunks = rng.standard_normal((n, samples_per_frame)).astype(np.float32) * 0.02
    return chunks


def _generate_imu(n: int) -> np.ndarray:
    """Generate realistic-looking IMU data: gravity on Z + small gyro noise."""
    rng  = np.random.default_rng()
    data = np.zeros((n, 10), dtype=np.float32)
    # Accel XYZ: mostly gravity on Z, slight motion
    data[:, 0] = rng.normal(0.0,  0.3, n).astype(np.float32)  # ax
    data[:, 1] = rng.normal(0.0,  0.3, n).astype(np.float32)  # ay
    data[:, 2] = rng.normal(9.81, 0.1, n).astype(np.float32)  # az (gravity)
    # Gyro XYZ: small angular velocities
    data[:, 3] = rng.normal(0.0, 0.05, n).astype(np.float32)  # gx
    data[:, 4] = rng.normal(0.0, 0.05, n).astype(np.float32)  # gy
    data[:, 5] = rng.normal(0.0, 0.05, n).astype(np.float32)  # gz
    # Quaternion XYZW: near identity with small perturbation
    data[:, 6] = rng.normal(0.0, 0.01, n).astype(np.float32)  # qx
    data[:, 7] = rng.normal(0.0, 0.01, n).astype(np.float32)  # qy
    data[:, 8] = rng.normal(0.0, 0.01, n).astype(np.float32)  # qz
    data[:, 9] = rng.normal(1.0, 0.01, n).astype(np.float32)  # qw (≈1)
    return data


def _generate_actions(n: int) -> np.ndarray:
    """Generate small head-pose delta actions."""
    rng     = np.random.default_rng()
    actions = np.zeros((n, 7), dtype=np.float32)
    actions[:, 0:3] = rng.normal(0, 0.01, (n, 3)).astype(np.float32)  # dx dy dz
    actions[:, 3:7] = rng.normal(0, 0.005, (n, 4)).astype(np.float32) # dq
    return actions


def _generate_timestamps(n: int, fps: int = 30, start: float = 0.0) -> np.ndarray:
    dt = 1.0 / fps
    return (start + np.arange(n, dtype=np.float64) * dt)


# ── Session writers ───────────────────────────────────────────────────────────

def _write_raw_session(session_dir: Path, n_frames: int, fps: int, sr: int) -> dict:
    """Write raw recorder-style files (video.mp4 simulated as CSV frames, audio.wav, imu.csv)."""
    session_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())
    start_time = 1700000000.0  # fixed base time for reproducibility

    # Write IMU CSV
    imu_hz  = 100
    n_imu   = int(n_frames / fps * imu_hz)
    imu_ts  = start_time + np.arange(n_imu) / imu_hz
    rng     = np.random.default_rng()
    noise   = rng.standard_normal((n_imu, 6)) * np.array([0.3, 0.3, 0.1, 0.05, 0.05, 0.05])
    imu_csv = session_dir / "imu.csv"
    with open(imu_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])
        for i in range(n_imu):
            w.writerow([
                imu_ts[i],
                noise[i, 0], noise[i, 1], 9.81 + noise[i, 2],
                noise[i, 3], noise[i, 4], noise[i, 5],
            ])

    # Write audio WAV
    samples_per_frame = sr // fps
    n_audio = n_frames * samples_per_frame
    audio   = (rng.standard_normal(n_audio) * 0.02 * 32767).astype(np.int16)
    wav_path = session_dir / "audio.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

    # Write video timestamps CSV (skip actual MP4 for offline test)
    frame_ts = start_time + np.arange(n_frames) / fps
    ts_csv = session_dir / "video_timestamps.csv"
    with open(ts_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp"])
        for i, t in enumerate(frame_ts):
            w.writerow([i, t])

    # Write metadata
    meta = {
        "session_id":        session_id,
        "start_time":        start_time,
        "end_time":          start_time + n_frames / fps,
        "video_fps":         fps,
        "audio_sample_rate": sr,
        "imu_hz":            imu_hz,
        "video_frames":      n_frames,
        "audio_samples":     n_audio,
        "imu_samples":       n_imu,
        "data_dir":          str(session_dir),
        "language":          "hi",
    }
    (session_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    return meta


def _write_hdf5_episode(
    output_dir: Path,
    session_idx: int,
    episode_idx: int,
    n_frames: int,
    fps: int,
    sr: int,
    instruction: str,
) -> Path:
    """Write a single LeRobot HDF5 episode directly."""
    ep_path = output_dir / f"episode_{episode_idx:06d}.hdf5"
    import h5py

    frames     = _generate_frames(n_frames)
    audio      = _generate_audio_chunks(n_frames, sr=sr, fps=fps)
    imu        = _generate_imu(n_frames)
    actions    = _generate_actions(n_frames)
    timestamps = _generate_timestamps(n_frames, fps=fps, start=float(session_idx * 1000))

    comp = dict(compression="gzip", compression_opts=4)
    with h5py.File(str(ep_path), "w") as f:
        obs = f.create_group("observation")
        img = obs.create_group("images")
        img.create_dataset("front_camera", data=frames, **comp)
        obs.create_dataset("audio", data=audio, **comp)
        obs.create_dataset("imu",   data=imu,   **comp)

        f.create_dataset("action",        data=actions)
        f.create_dataset("episode_index", data=np.full(n_frames, episode_idx, dtype=np.int64))
        f.create_dataset("frame_index",   data=np.arange(n_frames, dtype=np.int64))
        f.create_dataset("timestamp",     data=timestamps)

        dt = h5py.special_dtype(vlen=str)
        ld = f.create_dataset("language_instruction", (1,), dtype=dt)
        ld[0] = instruction

        meta_grp = f.create_group("metadata")
        meta_grp.attrs["episode_index"]   = episode_idx
        meta_grp.attrs["num_frames"]      = n_frames
        meta_grp.attrs["lerobot_version"] = "0.5"
        meta_grp.attrs["session_id"]      = f"synthetic-{session_idx:04d}"

    return ep_path


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def main(
    num_sessions: int = typer.Option(10, "--num-sessions", "-n",
                                     help="Number of synthetic sessions to generate."),
    frames_per_session: int = typer.Option(100, "--frames-per-session", "-f",
                                           help="Frames per session (= frames per episode for synthetic data)."),
    output_dir: str = typer.Option("/data/synthetic_episodes", "--output-dir", "-o",
                                   help="Output directory for HDF5 episodes.",
                                   envvar="DATA_DIR"),
    fps: int  = typer.Option(30, "--fps"),
    sr:  int  = typer.Option(16000, "--sample-rate"),
    also_raw: bool = typer.Option(False, "--also-raw",
                                  help="Also write raw recorder-style files (for pipeline testing)."),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Generate synthetic VLA training data for offline testing."""
    np.random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]India VLA Synthetic Data Generator[/bold cyan]")
    console.print(f"Sessions       : {num_sessions}")
    console.print(f"Frames/session : {frames_per_session}")
    console.print(f"Output dir     : {out}")
    console.print()

    raw_dir = out.parent / "synthetic_raw" if also_raw else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating episodes …", total=num_sessions)

        for i in range(num_sessions):
            instruction = _random_instruction()
            ep_path = _write_hdf5_episode(
                output_dir=out,
                session_idx=i,
                episode_idx=i,
                n_frames=frames_per_session,
                fps=fps,
                sr=sr,
                instruction=instruction,
            )

            if also_raw and raw_dir:
                sess_dir = raw_dir / f"session_{i:04d}"
                _write_raw_session(sess_dir, frames_per_session, fps, sr)

            progress.advance(task)

    # Write dataset_info.json
    import h5py, json

    first_ep_path = out / "episode_000000.hdf5"
    samples_per_frame = sr // fps

    info_dict = {
        "codebase_version": "0.5",
        "dataset_type":     "vla_egocentric",
        "dataset_name":     "india-pov-vla-synthetic",
        "fps":              fps,
        "total_episodes":   num_sessions,
        "total_frames":     num_sessions * frames_per_session,
        "languages":        ["hi", "en"],
        "features": {
            "observation.images.front_camera": {
                "dtype": "image", "shape": [224, 224, 3]},
            "observation.audio": {
                "dtype": "float32", "shape": [samples_per_frame]},
            "observation.imu": {
                "dtype": "float32", "shape": [10]},
            "action": {
                "dtype": "float32", "shape": [7]},
            "language_instruction": {"dtype": "string"},
        },
    }
    (out / "dataset_info.json").write_text(json.dumps(info_dict, indent=2, ensure_ascii=False))

    console.print()
    console.print(f"[bold green]Done![/bold green] Generated {num_sessions} episodes in [cyan]{out}[/cyan]")
    console.print(f"  Total frames : {num_sessions * frames_per_session}")
    console.print(f"  Dataset info : {out / 'dataset_info.json'}")


if __name__ == "__main__":
    app()

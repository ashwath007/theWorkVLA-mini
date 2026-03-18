"""
CLI for the HeadsetRecorder.

Usage
-----
    vla-record record --session-name my-session --duration 60
    vla-record stop
"""

from __future__ import annotations

import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .recorder import HeadsetRecorder

app = typer.Typer(
    name="vla-record",
    help="India Egocentric VLA — headset recorder CLI",
    add_completion=False,
)
console = Console()


def _make_stats_table(stats: dict) -> Table:
    """Render live recording stats as a Rich table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value",  style="green")

    elapsed = stats.get("elapsed_sec", 0)
    minutes, seconds = divmod(int(elapsed), 60)

    table.add_row("Session ID",   str(stats.get("session_id", "—")))
    table.add_row("Duration",     f"{minutes:02d}:{seconds:02d}")
    table.add_row("Video Frames", str(stats.get("video_frames", 0)))
    table.add_row("Audio Samples", str(stats.get("audio_samples", 0)))
    table.add_row("IMU Samples",  str(stats.get("imu_samples", 0)))
    return table


@app.command()
def record(
    session_name: Optional[str] = typer.Option(
        None, "--session-name", "-n",
        help="Human-readable session name (used as session_id prefix).",
    ),
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d",
        help="Recording duration in seconds. If omitted, press Ctrl+C to stop.",
    ),
    data_dir: str = typer.Option(
        "/data/sessions", "--data-dir",
        help="Root directory for storing session data.",
        envvar="DATA_DIR",
    ),
    video_device: int = typer.Option(0, "--video-device", help="Camera device index."),
    simulate_imu: bool = typer.Option(
        False, "--simulate-imu",
        help="Use simulated IMU (for development / testing).",
    ),
) -> None:
    """Start a new recording session."""
    session_id = f"{session_name}-{uuid.uuid4().hex[:8]}" if session_name else str(uuid.uuid4())

    recorder = HeadsetRecorder(
        data_root=data_dir,
        video_device=video_device,
        simulate_imu=simulate_imu if simulate_imu else None,
    )

    # Graceful Ctrl+C handler
    stop_flag = {"stop": False}

    def _handle_signal(signum, frame):
        stop_flag["stop"] = True
        console.print("\n[yellow]Interrupt received — stopping recording …[/yellow]")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        recorder.start(session_id=session_id)
        console.print(Panel(
            f"[bold green]Recording started[/bold green]\n"
            f"Session: [cyan]{session_id}[/cyan]\n"
            f"Output : [cyan]{data_dir}[/cyan]\n"
            f"{'Duration: ' + str(duration) + 's' if duration else 'Press Ctrl+C to stop'}",
            title="[bold]India VLA Recorder",
        ))

        start = time.time()
        with Live(console=console, refresh_per_second=4) as live:
            while not stop_flag["stop"]:
                if duration and (time.time() - start) >= duration:
                    break
                stats = recorder.stats
                live.update(Panel(_make_stats_table(stats), title="[bold]Live Stats"))
                time.sleep(0.25)

    finally:
        if recorder.is_recording:
            meta = recorder.stop()
            console.print(Panel(
                f"[bold green]Recording saved[/bold green]\n"
                f"Session : [cyan]{meta.session_id}[/cyan]\n"
                f"Location: [cyan]{meta.data_dir}[/cyan]\n"
                f"Frames  : {meta.video_frames}\n"
                f"Duration: {round((meta.end_time or 0) - meta.start_time, 1)}s",
                title="[bold]Session Complete",
            ))


@app.command()
def info(
    data_dir: str = typer.Option(
        "/data/sessions", "--data-dir", envvar="DATA_DIR",
    ),
) -> None:
    """List all recorded sessions in the data directory."""
    base = Path(data_dir)
    if not base.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        raise typer.Exit(1)

    sessions = sorted(base.rglob("metadata.json"))
    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    import json

    table = Table(title="Recorded Sessions", show_header=True)
    table.add_column("Session ID", style="cyan")
    table.add_column("Date/Time",  style="green")
    table.add_column("Frames",     justify="right")
    table.add_column("Duration",   justify="right")
    table.add_column("Path",       style="dim")

    for meta_path in sessions:
        try:
            meta = json.loads(meta_path.read_text())
            duration_s = (meta.get("end_time") or 0) - (meta.get("start_time") or 0)
            from datetime import datetime
            dt_str = datetime.fromtimestamp(meta.get("start_time", 0)).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                meta.get("session_id", "?"),
                dt_str,
                str(meta.get("video_frames", 0)),
                f"{duration_s:.1f}s",
                str(meta_path.parent),
            )
        except Exception as exc:
            table.add_row("?", "?", "?", "?", f"Error: {exc}")

    console.print(table)


if __name__ == "__main__":
    app()

"""
HFUploader: Upload LeRobot episode datasets to Hugging Face Hub.

Handles single-episode uploads, bulk directory uploads with progress, and
dataset card (model card) generation with India-VLA specific metadata.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


class HFUploader:
    """
    Uploads India VLA dataset episodes to the Hugging Face Hub.

    Parameters
    ----------
    repo_id : str
        HF repository ID, e.g., 'your-username/india-pov-vla-v1'.
    token : str, optional
        HF API token. Falls back to HF_TOKEN env var.
    private : bool
        Whether to create the repo as private.
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.private = private
        self.token   = token or os.environ.get("HF_TOKEN", "")

        if not self.token:
            logger.warning(
                "No HF_TOKEN found. Set the HF_TOKEN environment variable "
                "or pass token= to HFUploader."
            )

        self._api = None

    def _get_api(self):
        """Lazily initialise the HF Hub API."""
        if self._api is None:
            from huggingface_hub import HfApi, login
            if self.token:
                login(token=self.token, add_to_git_credential=False)
            self._api = HfApi()
        return self._api

    def _ensure_repo_exists(self, repo_type: str = "dataset") -> None:
        """Create the repository on HF Hub if it doesn't exist."""
        api = self._get_api()
        try:
            api.repo_info(repo_id=self.repo_id, repo_type=repo_type)
            logger.debug("Repository %s already exists.", self.repo_id)
        except Exception:
            api.create_repo(
                repo_id=self.repo_id,
                repo_type=repo_type,
                private=self.private,
                exist_ok=True,
            )
            logger.info("Created HF repository: %s", self.repo_id)

    # ── Single episode ────────────────────────────────────────────────────────────

    def upload_episode(
        self,
        episode_path: str,
        split: str = "train",
        path_in_repo: Optional[str] = None,
    ) -> str:
        """
        Upload a single episode HDF5 file to the HF Hub.

        Parameters
        ----------
        episode_path : str  local path to .hdf5 file
        split : str         dataset split ('train', 'validation', 'test')
        path_in_repo : str  optional explicit path in repo

        Returns
        -------
        str  URL of uploaded file.
        """
        api = self._get_api()
        self._ensure_repo_exists()

        ep_path_obj = Path(episode_path)
        repo_path   = path_in_repo or f"data/{split}/{ep_path_obj.name}"

        url = api.upload_file(
            path_or_fileobj=episode_path,
            path_in_repo=repo_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        logger.info("Uploaded %s → %s", episode_path, url)
        return url

    # ── Bulk upload ───────────────────────────────────────────────────────────────

    def upload_dataset(
        self,
        data_dir: str,
        repo_id: Optional[str] = None,
        split: str = "train",
        file_glob: str = "*.hdf5",
    ) -> List[str]:
        """
        Upload all HDF5 episode files from data_dir to the HF Hub.

        Parameters
        ----------
        data_dir : str    local directory containing .hdf5 files
        repo_id  : str    override self.repo_id for this upload
        split    : str    train / validation / test
        file_glob: str    glob pattern for episode files

        Returns
        -------
        list of str  URLs of uploaded files.
        """
        if repo_id:
            self.repo_id = repo_id

        base  = Path(data_dir)
        files = sorted(base.rglob(file_glob))
        if not files:
            logger.warning("No %s files found in %s", file_glob, data_dir)
            return []

        self._ensure_repo_exists()
        urls: List[str] = []

        for ep_path in tqdm(files, desc="Uploading episodes", unit="file"):
            try:
                url = self.upload_episode(str(ep_path), split=split)
                urls.append(url)
            except Exception as exc:
                logger.error("Failed to upload %s: %s", ep_path, exc)

        # Upload dataset_info.json if present
        info_path = base / "dataset_info.json"
        if info_path.exists():
            try:
                api = self._get_api()
                api.upload_file(
                    path_or_fileobj=str(info_path),
                    path_in_repo="dataset_info.json",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
                logger.info("Uploaded dataset_info.json")
            except Exception as exc:
                logger.warning("Could not upload dataset_info.json: %s", exc)

        logger.info("Uploaded %d / %d files to %s", len(urls), len(files), self.repo_id)
        return urls

    # ── Dataset card ─────────────────────────────────────────────────────────────

    def create_dataset_card(
        self,
        repo_id: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate and push a dataset model card (README.md) to the HF Hub.

        Parameters
        ----------
        repo_id : str   override self.repo_id
        stats   : dict  from HDF5Store.get_session_stats()

        Returns
        -------
        str  URL of the created/updated README.md.
        """
        repo = repo_id or self.repo_id
        stats = stats or {}

        total_episodes = stats.get("total_episodes", "N/A")
        total_frames   = stats.get("total_frames", "N/A")
        duration_sec   = stats.get("total_duration_sec", 0)
        duration_hr    = round(duration_sec / 3600, 2) if isinstance(duration_sec, (int, float)) else "N/A"
        languages      = stats.get("languages", ["hi", "en"])

        card_content = f"""---
language:
{chr(10).join(f"- {lang}" for lang in languages)}
license: apache-2.0
task_categories:
- robotics
- visual-question-answering
tags:
- vla
- lerobot
- egocentric
- india
- hindi
- robotics
- multimodal
dataset_info:
  features:
    - name: observation.images.front_camera
      dtype: image
    - name: observation.audio
      dtype: float32
    - name: observation.imu
      dtype: float32
    - name: action
      dtype: float32
    - name: language_instruction
      dtype: string
  splits:
    - name: train
      num_examples: {total_episodes}
---

# India Egocentric VLA Dataset

**India Egocentric Vision-Language-Action (VLA) dataset** captured from first-person
headset recordings of workers performing real-world tasks in Indian industrial and
domestic environments.

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Episodes | {total_episodes} |
| Total Frames | {total_frames} |
| Total Duration | {duration_hr} hours |
| Languages | {", ".join(languages)} |
| Image Resolution | 224×224 RGB |
| Video FPS | 30 |
| Audio Sample Rate | 16 kHz |
| IMU Rate | 100 Hz (interpolated to 30 fps) |
| Action Space | 7-DOF head pose delta |

## Modalities

- **Video**: Egocentric front-facing camera at 224×224 resolution, 30 fps
- **Audio**: Mono 16 kHz WAV, chunk-aligned to video frames
- **IMU**: 6-DOF accelerometer + gyroscope + quaternion orientation
- **Language**: Hindi and English instructions (code-mixed), transcribed via Whisper

## Data Format (LeRobot v0.5)

Each HDF5 episode contains:
```
/observation/images/front_camera   (T, 224, 224, 3) uint8
/observation/audio                 (T, audio_samples) float32
/observation/imu                   (T, 10) float32
/action                            (T, 7) float32
/language_instruction              string (Hindi/English)
/episode_index                     (T,) int64
/frame_index                       (T,) int64
/timestamp                         (T,) float64
```

## Privacy

All faces visible in recordings are automatically blurred using OpenCV Haar cascade
detection before the dataset is published.

## License

Apache 2.0 — see LICENSE file.

## Citation

```bibtex
@dataset{{india_vla_2024,
  title     = {{India Egocentric VLA Dataset}},
  author    = {{India VLA Team}},
  year      = {{2024}},
  publisher = {{Hugging Face}},
  url       = {{https://huggingface.co/datasets/{repo}}}
}}
```
"""

        api = self._get_api()
        self._ensure_repo_exists()

        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(card_content)
            tmp_path = tmp.name

        try:
            url = api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
            )
        finally:
            os.unlink(tmp_path)

        logger.info("Dataset card pushed to %s", url)
        return url

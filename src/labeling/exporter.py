"""
LabelExporter: Convert LabelStudio annotations → LeRobot HDF5 episodes.

Flow:
    LabelStudio annotations (JSON)
        → parse action_type, instruction, bboxes, quality
        → match to segment HDF5 episodes
        → overwrite language_instruction + add label metadata
        → write final labeled_episodes/

Usage:
    exporter = LabelExporter(episodes_dir="sessions/XYZ/episodes")
    exporter.export(annotations, output_dir="sessions/XYZ/labeled_episodes")
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class LabelExporter:
    """
    Applies human-verified annotations from LabelStudio to episode HDF5 files.

    Parameters
    ----------
    episodes_dir : str | Path
        Directory containing episode_NNNNNN.h5 files from the pipeline.
    """

    def __init__(self, episodes_dir: str | Path) -> None:
        self.episodes_dir = Path(episodes_dir)

    # ── Public ────────────────────────────────────────────────────────────────

    def export(
        self,
        annotations: List[Dict],
        output_dir: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply annotations and write labeled episodes.

        Parameters
        ----------
        annotations : list of annotation dicts from LabelStudioClient.pull_annotations()
        output_dir : destination directory. Defaults to episodes_dir/../labeled_episodes
        overwrite : if True, overwrite existing labeled episodes

        Returns
        -------
        dict with keys: total, updated, skipped, errors
        """
        if output_dir is None:
            output_dir = self.episodes_dir.parent / "labeled_episodes"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build index: segment_index → annotation
        ann_index = self._build_annotation_index(annotations)

        # Discover episode files
        ep_paths = sorted(self.episodes_dir.glob("episode_*.h5"))
        if not ep_paths:
            logger.warning("No episode files found in %s", self.episodes_dir)
            return {"total": 0, "updated": 0, "skipped": 0, "errors": []}

        total    = 0
        updated  = 0
        skipped  = 0
        errors: List[str] = []

        for ep_path in ep_paths:
            total += 1
            ep_idx = self._parse_episode_index(ep_path.stem)
            dest   = output_dir / ep_path.name

            if dest.exists() and not overwrite:
                skipped += 1
                continue

            ann = ann_index.get(ep_idx)
            try:
                shutil.copy2(ep_path, dest)
                if ann:
                    self._apply_annotation(dest, ann)
                    updated += 1
                else:
                    skipped += 1
                    logger.debug("No annotation for episode %d — copied as-is", ep_idx)
            except Exception as exc:
                errors.append(f"episode_{ep_idx}: {exc}")
                logger.error("Error exporting episode %d: %s", ep_idx, exc)

        # Write annotation summary alongside episodes
        summary = {
            "total": total, "updated": updated,
            "skipped": skipped, "errors": errors,
        }
        (output_dir / "export_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        # Write label stats
        stats = self._compute_label_stats(ann_index)
        (output_dir / "label_stats.json").write_text(
            json.dumps(stats, indent=2)
        )

        logger.info("Export done: %d updated, %d skipped, %d errors", updated, skipped, len(errors))
        return summary

    def export_from_file(
        self,
        annotations_json_path: str,
        output_dir: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Convenience: load annotations from a JSON file and export."""
        with open(annotations_json_path) as f:
            annotations = json.load(f)
        return self.export(annotations, output_dir, overwrite)

    def to_jsonl(
        self,
        annotations: List[Dict],
        output_path: str,
    ) -> int:
        """
        Write annotations as JSONL for downstream training scripts.

        Each line: {episode_path, action_type, scenario, instruction_hi, instruction_en, quality, bboxes}
        """
        ep_paths = {self._parse_episode_index(p.stem): p for p in self.episodes_dir.glob("episode_*.h5")}
        ann_index = self._build_annotation_index(annotations)

        count = 0
        with open(output_path, "w", encoding="utf-8") as out:
            for ep_idx, ann in ann_index.items():
                ep_path = ep_paths.get(ep_idx)
                record  = {
                    "episode_path":    str(ep_path) if ep_path else None,
                    "episode_index":   ep_idx,
                    "action_type":     ann.get("action_type"),
                    "scenario":        ann.get("scenario"),
                    "instruction_hi":  ann.get("instruction_hi", ""),
                    "instruction_en":  ann.get("instruction_en", ""),
                    "quality":         ann.get("quality", 2),
                    "bboxes":          ann.get("bboxes", []),
                    "annotator":       ann.get("annotator"),
                    "created_at":      ann.get("created_at"),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        logger.info("Wrote %d annotation records to %s", count, output_path)
        return count

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_annotation_index(self, annotations: List[Dict]) -> Dict[int, Dict]:
        """
        Build {segment_index → parsed_annotation} from raw LS annotation list.
        """
        index: Dict[int, Dict] = {}
        for ann in annotations:
            seg_idx = ann.get("segment_index")
            if seg_idx is None:
                # Fall back to frame_index-based lookup
                seg_idx = ann.get("task_id", -1)

            parsed = self._parse_annotation(ann)
            if seg_idx is not None:
                index[int(seg_idx)] = parsed

        return index

    def _parse_annotation(self, ann: Dict) -> Dict:
        """Extract structured fields from one LabelStudio annotation."""
        result: List[Dict] = ann.get("annotation", [])

        action_type    = ""
        scenario       = ""
        instruction_hi = ""
        instruction_en = ""
        quality        = 2
        bboxes: List[Dict] = []

        for r in result:
            from_name = r.get("from_name", "")
            val       = r.get("value", {})

            if from_name == "action_type":
                choices = val.get("choices", [])
                action_type = choices[0] if choices else ""

            elif from_name == "scenario":
                choices = val.get("choices", [])
                scenario = choices[0] if choices else ""

            elif from_name == "instruction_hi":
                texts = val.get("text", [])
                instruction_hi = texts[0] if texts else ""

            elif from_name == "instruction_en":
                texts = val.get("text", [])
                instruction_en = texts[0] if texts else ""

            elif from_name == "quality":
                quality = int(val.get("rating", 2))

            elif from_name == "objects":
                bboxes.append({
                    "label":  val.get("rectanglelabels", ["unknown"])[0],
                    "x":      val.get("x", 0),
                    "y":      val.get("y", 0),
                    "width":  val.get("width", 0),
                    "height": val.get("height", 0),
                })

        return {
            "action_type":    action_type,
            "scenario":       scenario,
            "instruction_hi": instruction_hi,
            "instruction_en": instruction_en,
            "quality":        quality,
            "bboxes":         bboxes,
            "annotator":      ann.get("annotator"),
            "created_at":     ann.get("created_at"),
        }

    def _apply_annotation(self, hdf5_path: Path, ann: Dict) -> None:
        """
        Write annotation fields into an existing HDF5 episode file.
        Updates: language_instruction, and adds /labels/ group with metadata.
        """
        with h5py.File(str(hdf5_path), "a") as f:
            # Update language instruction with human-verified version
            instr = ann.get("instruction_hi") or ann.get("instruction_en") or ""
            if instr:
                if "language_instruction" in f:
                    del f["language_instruction"]
                f.create_dataset(
                    "language_instruction",
                    data=instr.encode("utf-8"),
                )

            # Add labels group
            if "labels" in f:
                del f["labels"]
            grp = f.create_group("labels")
            grp.attrs["action_type"]    = ann.get("action_type", "")
            grp.attrs["scenario"]       = ann.get("scenario", "")
            grp.attrs["instruction_hi"] = ann.get("instruction_hi", "")
            grp.attrs["instruction_en"] = ann.get("instruction_en", "")
            grp.attrs["quality"]        = int(ann.get("quality", 2))
            grp.attrs["annotator"]      = str(ann.get("annotator") or "")
            grp.attrs["created_at"]     = str(ann.get("created_at") or "")
            grp.attrs["is_human_labeled"] = True

            # Store bboxes as a JSON string in an attribute
            bboxes = ann.get("bboxes", [])
            grp.attrs["bboxes_json"] = json.dumps(bboxes)

        logger.debug("Applied annotation to %s", hdf5_path.name)

    def _compute_label_stats(self, ann_index: Dict[int, Dict]) -> Dict:
        """Aggregate stats over all annotations."""
        from collections import Counter
        actions   = Counter(a.get("action_type", "")  for a in ann_index.values())
        scenarios = Counter(a.get("scenario", "")     for a in ann_index.values())
        qualities = Counter(int(a.get("quality", 2))  for a in ann_index.values())
        return {
            "total_annotated": len(ann_index),
            "action_types":    dict(actions),
            "scenarios":       dict(scenarios),
            "quality_dist":    dict(qualities),
        }

    @staticmethod
    def _parse_episode_index(stem: str) -> int:
        """Parse episode index from filename stem like 'episode_000042'."""
        try:
            return int(stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

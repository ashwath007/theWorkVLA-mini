"""
LabelStudioClient: Create projects, push tasks, and pull annotations.

Works with Label Studio running locally (docker compose up) or remotely.
Docs: https://labelstud.io/guide/api.html

Usage:
    client = LabelStudioClient("http://localhost:8080", api_key="your-key")
    project_id = client.get_or_create_project("india-vla-delivery")
    task_ids = client.push_frames(project_id, frames, session_id, frame_indices)
    annotations = client.pull_annotations(project_id)
"""

from __future__ import annotations

import base64
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Label Studio project config for egocentric VLA labeling
VLA_LABEL_CONFIG = """
<View>
  <Header value="Label the action and objects in this frame"/>
  <Image name="image" value="$image" maxWidth="800px"/>

  <Choices name="action_type" toName="image" choice="single" showInLine="true">
    <Header value="Action Type"/>
    <Choice value="pick_up"      label="Pick Up"/>
    <Choice value="put_down"     label="Put Down"/>
    <Choice value="hand_over"    label="Hand Over"/>
    <Choice value="open_door"    label="Open Door"/>
    <Choice value="close_door"   label="Close Door"/>
    <Choice value="navigate"     label="Navigate/Walk"/>
    <Choice value="driving"      label="Driving"/>
    <Choice value="idle"         label="Idle/Stationary"/>
    <Choice value="scan_barcode" label="Scan Barcode"/>
    <Choice value="pay_collect"  label="Pay/Collect Payment"/>
    <Choice value="other"        label="Other"/>
  </Choices>

  <Choices name="scenario" toName="image" choice="single" showInLine="true">
    <Header value="Scenario"/>
    <Choice value="food_delivery"   label="Food Delivery"/>
    <Choice value="quick_commerce"  label="Quick Commerce"/>
    <Choice value="driving"         label="Driving"/>
    <Choice value="warehouse"       label="Warehouse"/>
    <Choice value="kitchen"         label="Kitchen"/>
    <Choice value="street"          label="Street/Outdoor"/>
    <Choice value="office"          label="Office"/>
    <Choice value="other"           label="Other"/>
  </Choices>

  <TextArea name="instruction_hi" toName="image"
            placeholder="हिंदी में निर्देश लिखें (Write instruction in Hindi)"
            maxSubmissions="1" editable="true"/>
  <TextArea name="instruction_en" toName="image"
            placeholder="English instruction (optional)"
            maxSubmissions="1" editable="true"/>

  <RectangleLabels name="objects" toName="image">
    <Label value="package"      background="#FF6B6B"/>
    <Label value="door"         background="#4ECDC4"/>
    <Label value="person"       background="#45B7D1"/>
    <Label value="vehicle"      background="#96CEB4"/>
    <Label value="phone"        background="#FFEAA7"/>
    <Label value="food_bag"     background="#DDA0DD"/>
    <Label value="hand"         background="#98D8C8"/>
    <Label value="road_sign"    background="#F7DC6F"/>
    <Label value="building"     background="#BB8FCE"/>
  </RectangleLabels>

  <Rating name="quality" toName="image" maxRating="3" icon="star"
          toName="image" hint="1=poor, 2=ok, 3=good"/>
</View>
"""


class LabelStudioClient:
    """
    Thin client for the Label Studio REST API.

    Parameters
    ----------
    url : str
        Label Studio base URL, e.g. "http://localhost:8080".
    api_key : str
        User API key from Label Studio Account → Access Token.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(self, url: str, api_key: str, timeout: int = 30) -> None:
        self.url     = url.rstrip("/")
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type":  "application/json",
        }
        self.timeout = timeout
        self._session = self._build_session()

    def _build_session(self):  # type: ignore[return]
        try:
            import requests
            s = requests.Session()
            s.headers.update(self.headers)
            return s
        except ImportError:
            logger.error("requests library not installed — LabelStudioClient will not work")
            return None

    # ── Projects ──────────────────────────────────────────────────────────────

    def list_projects(self) -> List[Dict]:
        resp = self._get("/api/projects")
        return resp.get("results", [])

    def get_or_create_project(
        self,
        title: str,
        label_config: str = VLA_LABEL_CONFIG,
        description: str  = "India VLA egocentric labeling",
    ) -> int:
        """
        Return existing project id matching `title`, or create a new one.

        Returns
        -------
        project_id : int
        """
        for project in self.list_projects():
            if project.get("title") == title:
                pid = int(project["id"])
                logger.info("Found existing project '%s' (id=%d)", title, pid)
                return pid

        payload = {
            "title":        title,
            "description":  description,
            "label_config": label_config,
        }
        resp = self._post("/api/projects", payload)
        pid  = int(resp["id"])
        logger.info("Created project '%s' (id=%d)", title, pid)
        return pid

    def get_project(self, project_id: int) -> Dict:
        return self._get(f"/api/projects/{project_id}")

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def push_frames(
        self,
        project_id: int,
        frames: List[np.ndarray],
        session_id: str,
        frame_indices: Optional[List[int]] = None,
        meta: Optional[Dict] = None,
        batch_size: int = 50,
    ) -> List[int]:
        """
        Upload frames as labeling tasks.

        Each frame is base64-encoded as a JPEG and embedded directly in the
        task JSON (no external storage needed for prototyping).

        Parameters
        ----------
        frames : list of (H, W, 3) uint8 numpy arrays
        session_id : str
        frame_indices : optional list of original frame indices
        meta : extra metadata to attach to each task
        batch_size : number of tasks per API call

        Returns
        -------
        list of created task ids
        """
        if frame_indices is None:
            frame_indices = list(range(len(frames)))

        tasks = []
        for i, (frame, fidx) in enumerate(zip(frames, frame_indices)):
            img_b64 = self._frame_to_b64(frame)
            task_data = {
                "image":         f"data:image/jpeg;base64,{img_b64}",
                "session_id":    session_id,
                "frame_index":   fidx,
                **(meta or {}),
            }
            tasks.append({"data": task_data})

        task_ids: List[int] = []
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start: batch_start + batch_size]
            resp  = self._post(f"/api/projects/{project_id}/import", batch)
            if isinstance(resp, list):
                task_ids.extend(t.get("id") for t in resp if "id" in t)
            elif isinstance(resp, dict) and "task_ids" in resp:
                task_ids.extend(resp["task_ids"])
            time.sleep(0.1)  # gentle rate-limit

        logger.info("Pushed %d tasks to project %d", len(task_ids), project_id)
        return task_ids

    def push_segment_tasks(
        self,
        project_id: int,
        frames: List[np.ndarray],
        segments: List[Dict],
        session_id: str,
        auto_labels: Optional[List[Dict]] = None,
    ) -> List[int]:
        """
        Push one representative frame per action segment as a task.

        Parameters
        ----------
        segments : list of {start_frame, end_frame, confidence}
        auto_labels : pre-annotations from AutoLabeler (one per segment)
        """
        task_ids = []
        for seg_idx, seg in enumerate(segments):
            mid_frame  = (seg["start_frame"] + seg["end_frame"]) // 2
            mid_frame  = min(mid_frame, len(frames) - 1)
            frame      = frames[mid_frame]
            img_b64    = self._frame_to_b64(frame)

            task_data = {
                "image":         f"data:image/jpeg;base64,{img_b64}",
                "session_id":    session_id,
                "segment_index": seg_idx,
                "start_frame":   seg["start_frame"],
                "end_frame":     seg["end_frame"],
                "confidence":    seg.get("confidence", 0.0),
                "frame_index":   mid_frame,
            }
            task = {"data": task_data}

            # Attach auto-label as prediction (pre-annotation)
            if auto_labels and seg_idx < len(auto_labels):
                al = auto_labels[seg_idx]
                task["predictions"] = [self._build_prediction(al)]

            resp = self._post(f"/api/projects/{project_id}/import", [task])
            if isinstance(resp, list) and resp:
                task_ids.append(resp[0].get("id"))
            elif isinstance(resp, dict) and "task_ids" in resp:
                task_ids.extend(resp["task_ids"])

        logger.info("Pushed %d segment tasks to project %d", len(task_ids), project_id)
        return task_ids

    def get_tasks(self, project_id: int, page_size: int = 100) -> List[Dict]:
        """Return all tasks (with annotations) for a project."""
        tasks: List[Dict] = []
        page = 1
        while True:
            resp = self._get(
                f"/api/tasks",
                params={"project": project_id, "page": page, "page_size": page_size},
            )
            batch = resp.get("tasks") or resp.get("results") or []
            tasks.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return tasks

    # ── Annotations ──────────────────────────────────────────────────────────

    def pull_annotations(self, project_id: int) -> List[Dict]:
        """
        Return all completed annotations for a project.

        Returns
        -------
        list of annotation dicts with task metadata included.
        """
        tasks       = self.get_tasks(project_id)
        annotations = []
        for task in tasks:
            for ann in task.get("annotations", []):
                if ann.get("was_cancelled"):
                    continue
                annotations.append({
                    "task_id":       task["id"],
                    "session_id":    task["data"].get("session_id"),
                    "frame_index":   task["data"].get("frame_index"),
                    "segment_index": task["data"].get("segment_index"),
                    "start_frame":   task["data"].get("start_frame"),
                    "end_frame":     task["data"].get("end_frame"),
                    "annotation":    ann["result"],
                    "annotator":     ann.get("completed_by"),
                    "created_at":    ann.get("created_at"),
                })
        logger.info("Pulled %d annotations from project %d", len(annotations), project_id)
        return annotations

    def export_annotations(self, project_id: int, fmt: str = "JSON") -> bytes:
        """Export annotations in the given format (JSON, CSV, YOLO, COCO…)."""
        resp = self._get(f"/api/projects/{project_id}/export", params={"exportType": fmt}, raw=True)
        return resp

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _frame_to_b64(frame: np.ndarray, quality: int = 75) -> str:
        """Convert a numpy uint8 frame to base64 JPEG string."""
        try:
            import cv2
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except ImportError:
            # Fallback: use PIL
            from PIL import Image
            img = Image.fromarray(frame)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _build_prediction(auto_label: Dict) -> Dict:
        """Convert AutoLabeler output to Label Studio prediction format."""
        result = []
        if auto_label.get("action_type"):
            result.append({
                "from_name": "action_type",
                "to_name":   "image",
                "type":      "choices",
                "value":     {"choices": [auto_label["action_type"]]},
            })
        if auto_label.get("scenario"):
            result.append({
                "from_name": "scenario",
                "to_name":   "image",
                "type":      "choices",
                "value":     {"choices": [auto_label["scenario"]]},
            })
        if auto_label.get("instruction_hi"):
            result.append({
                "from_name": "instruction_hi",
                "to_name":   "image",
                "type":      "textarea",
                "value":     {"text": [auto_label["instruction_hi"]]},
            })
        return {"result": result, "score": auto_label.get("confidence", 0.5)}

    def _get(self, path: str, params: Optional[Dict] = None, raw: bool = False):
        if self._session is None:
            raise RuntimeError("requests not installed")
        resp = self._session.get(
            f"{self.url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.content if raw else resp.json()

    def _post(self, path: str, payload: Any):
        if self._session is None:
            raise RuntimeError("requests not installed")
        resp = self._session.post(
            f"{self.url}{path}",
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

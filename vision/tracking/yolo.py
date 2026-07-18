from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from vision.common.logging import get_logger
from vision.common.paths import model_path as resolve_model_path
from vision.common.types import Detection, Track
from vision.tracking.base import ObjectTracker

logger = get_logger("vision.tracking.yolo", log_file="vision_tracking.log")

IOU_MATCH_THRESHOLD = 0.3
MAX_MISSED_FRAMES = 30


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


class _PersistentTrack:
    __slots__ = ("track_id", "class_id", "class_name", "bbox", "missed")

    def __init__(self, track_id, class_id, class_name, bbox):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = bbox
        self.missed = 0


class YOLOTracker(ObjectTracker):
    """Multi-object tracker with stable, self-managed track IDs.

    Detection is done by a YOLO ``model.predict()`` forward. Track identity is
    kept stable across frames by our own IoU-based association: a new detection
    is matched to the closest persisted track of the same class; if none
    overlaps enough, a new stable ``track_id`` is issued. This avoids the ID
    flicker that occurs when relying on the underlying tracker's re-ID / buffer
    resets, and needs only a single model forward per frame.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        tracker_config: str = "botsort.yaml",
        confidence: float = 0.4,
        image_size: int = 320,
    ):
        self.model = YOLO(str(model_path or resolve_model_path("yolo11n.pt")))
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.image_size = image_size
        self._next_id = 1
        self._active: dict[int, _PersistentTrack] = {}
        logger.info(
            "YOLOTracker ready (model=%s conf=%.2f imgsz=%d)",
            self.model.model_name if hasattr(self.model, "model_name") else model_path,
            confidence,
            image_size,
        )

    def _predict(self, frame) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        detections: list[Detection] = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                Detection(
                    class_id=cls,
                    class_name=self.model.names[cls],
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def _associate(self, detections: list[Detection]) -> list[Track]:
        tracks: list[Track] = []
        used: set[int] = set()

        for det in detections:
            best_id = None
            best_iou = IOU_MATCH_THRESHOLD
            for tid, pt in self._active.items():
                if tid in used or pt.class_id != det.class_id:
                    continue
                iou = _iou(det.bbox, pt.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1

            used.add(best_id)
            self._active[best_id] = _PersistentTrack(
                best_id, det.class_id, det.class_name, det.bbox
            )
            tracks.append(
                Track(
                    track_id=best_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )

        # Age out tracks that were not matched this frame.
        for tid in list(self._active.keys()):
            if tid not in used:
                self._active[tid].missed += 1
                if self._active[tid].missed > MAX_MISSED_FRAMES:
                    del self._active[tid]

        logger.debug("Tracked %d objects (%d persistent)", len(tracks), len(self._active))
        return tracks

    def update(
        self,
        frame,
        detections: list[Detection] | None = None,
    ) -> list[Track]:
        if detections is None:
            detections = self._predict(frame)
        return self._associate(detections)

    def detect_and_track(self, frame) -> tuple[list[Detection], list[Track]]:
        """Single forward returning both detections and stable-ID tracks."""
        detections = self._predict(frame)
        tracks = self._associate(detections)
        return detections, tracks



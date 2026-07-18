from __future__ import annotations

from pathlib import Path
from typing import Optional

from vision.common.client import VisionClient
from vision.common.logging import get_logger
from vision.common.types import Detection, FrameResult, Track
from vision.tracking.base import ObjectTracker
from vision.tracking.factory import create_tracker

logger = get_logger("vision.tracking.client", log_file="vision_tracking.log")


class TrackingClient(VisionClient):
    """Associates detections across frames into stable tracks (extends FrameResult).

    When the underlying tracker supports a combined detect+track forward
    (``detect_and_track``), this client is the single YOLO pass for the frame
    and also fills ``result.detections`` — avoiding a redundant detection stage.

    Single entry point used by ``tracking/__main__.py`` (CLI) and by the
    orchestrating pipeline.
    """

    name = "tracking"

    def __init__(
        self,
        tracker_name: str = "yolo",
        model_path: str | Path | None = None,
        tracker_config: str = "botsort.yaml",
        confidence: float = 0.4,
        image_size: int = 320,
    ):
        logger.info(
            "Initializing TrackingClient (tracker=%s conf=%.2f imgsz=%d)",
            tracker_name,
            confidence,
            image_size,
        )
        self.tracker: ObjectTracker = create_tracker(
            tracker_name,
            model_path=model_path,
            tracker_config=tracker_config,
            confidence=confidence,
            image_size=image_size,
        )
        self._combined = hasattr(self.tracker, "detect_and_track")

    def track(self, frame, detections=None) -> list[Track]:
        tracks = self.tracker.update(frame, detections)
        logger.debug("Tracked %d objects", len(tracks))
        return tracks

    def run(self, frame, result: Optional[FrameResult] = None) -> FrameResult:
        if result is None:
            result = FrameResult(frame=frame)

        if self._combined and not result.detections:
            # One YOLO pass yields both detections and tracks.
            detections, tracks = self.tracker.detect_and_track(frame)
            result.detections = detections
        else:
            logger.debug("Tracking stage: %d detections in", len(result.detections))
            tracks = self.track(frame, result.detections)
        result.tracks = tracks
        return result

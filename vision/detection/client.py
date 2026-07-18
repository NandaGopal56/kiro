from __future__ import annotations

from typing import Optional

from shared.logging import get_logger
from vision.common.paths import model_path
from vision.common.types import Detection, FrameResult
from vision.detection.base import ObjectDetector
from vision.detection.factory import create_detector

logger = get_logger("vision.detection.client", log_file="vision_detection.log")

DEFAULT_MODEL_PATH = model_path("yolo11n.pt")


class DetectionClient:
    """Runs object detection on a frame and extends a FrameResult.

    Standalone entry point used by ``detection/__main__.py`` (CLI) and composed
    by ``VisionPipeline`` — the only place that joins vision modules together.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.4,
        image_size: int = 320,
    ):
        self.detector: ObjectDetector = create_detector("yolo", model_path)
        if hasattr(self.detector, "confidence"):
            self.detector.confidence = confidence
        if hasattr(self.detector, "image_size"):
            self.detector.image_size = image_size

    def detect(self, frame) -> list[Detection]:
        detections = self.detector.detect(frame)
        logger.debug("Detected %d objects", len(detections))
        return detections

    def run(self, frame, result: Optional[FrameResult] = None) -> FrameResult:
        if result is None:
            result = FrameResult(frame=frame)
        result.detections = self.detect(frame)
        return result

from pathlib import Path

from ultralytics import YOLO

from vision.common.types import Detection
from vision.detection.base import ObjectDetector


class YOLODetector(ObjectDetector):

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.4,
        image_size: int = 320,
    ):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame) -> list[Detection]:

        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )

        detections = []

        for box in results[0].boxes:

            cls = int(box.cls[0])

            detections.append(
                Detection(
                    class_id=cls,
                    class_name=self.model.names[cls],
                    confidence=float(box.conf[0]),
                    bbox=tuple(map(int, box.xyxy[0])),
                )
            )

        return detections
from pathlib import Path

from vision.detection.yolo import YOLODetector


def create_detector(
    detector: str,
    model_path: str | Path,
):
    detector = detector.lower()

    if detector == "yolo":
        return YOLODetector(model_path)

    raise ValueError(f"Unsupported detector: {detector}")
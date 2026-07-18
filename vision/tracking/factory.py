from vision.tracking.yolo import YOLOTracker


def create_tracker(
    name: str,
    model_path=None,
    tracker_config: str = "botsort.yaml",
    confidence: float = 0.4,
    image_size: int = 320,
):
    name = name.lower()

    if name in {"yolo", "botsort", "bytetrack", "yolo-track"}:
        return YOLOTracker(
            model_path=model_path,
            tracker_config=trajectory_config(name, tracker_config),
            confidence=confidence,
            image_size=image_size,
        )

    raise ValueError(f"Unsupported tracker: {name}")


def trajectory_config(name: str, default: str) -> str:
    """Map the requested tracker name to an Ultralytics tracker config."""
    if name == "bytetrack":
        return "bytetrack.yaml"
    return default  # botsort.yaml

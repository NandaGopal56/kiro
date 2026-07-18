# vision/detection

Object detection module. Implements the `VisionClient` interface so it can run
standalone via `__main__` or be composed by `VisionPipeline`.

## Files

- **`base.py`** — `ObjectDetector` ABC with `detect(frame) -> list[Detection]`.
- **`factory.py`** — `create_detector("yolo", model_path)` builder.
- **`yolo.py`** — `YOLODetector` using `ultralytics` YOLO (downloads
  `yolo11m.pt` into the root `.models/` on first use if absent).
- **`client.py`** — `DetectionClient(VisionClient)`, the single entry point.
  Extends a `FrameResult` with `detections`.
- **`__main__.py`** — standalone CLI.

## Standalone CLI

```bash
uv run -m vision.detection --confidence 0.4 --image-size 320 --max-frames 200
```

| Flag           | Default | Description                  |
|----------------|---------|------------------------------|
| `--confidence` | `0.4`   | Detection confidence threshold |
| `--image-size` | `320`   | YOLO inference size          |
| `--max-frames` | `0`     | Stop after N frames (`0` = unlimited) |

## Library

```python
from vision.detection.client import DetectionClient
from vision.common.types import FrameResult

client = DetectionClient()                 # uses .models/yolo11m.pt
result: FrameResult = client.run(frame)
print(result.detections)                   # list[Detection]
```

## Extending

Add a new detector by subclassing `ObjectDetector`, then registering it in
`factory.py`. `client.py` and `__main__.py` need no changes.

# vision/tracking

Object tracking module. Associates detections across frames into stable track
IDs using a real multi-object tracker. Runs standalone (`__main__`) or is
composed by `VisionPipeline` (the only place modules are joined).

## Files

- **`base.py`** — `ObjectTracker` ABC with
  `update(frame, detections) -> list[Track]`.
- **`factory.py`** — `create_tracker(name, ...)` builder. Accepts
  `yolo` / `botsort` (BoT-SORT) and `bytetrack` (ByteTrack).
- **`yolo.py`** — `YOLOTracker`, a real tracker built on Ultralytics'
  `model.track()` with `persist=True`, so the same physical object keeps a
  stable `track_id` across frames. Class labels are read from the detection
  stage's model names.
- **`client.py`** — `TrackingClient`, the standalone entry point. Reads
  `result.detections` and extends `FrameResult` with `tracks`.
- **`__main__.py`** — standalone CLI (runs detection + tracking).

## Standalone CLI

```bash
uv run -m vision.tracking --tracker yolo --confidence 0.4 --max-frames 200
```

| Flag           | Default | Description                  |
|----------------|---------|------------------------------|
| `--tracker`    | `yolo`  | Tracker backend (`yolo`/`botsort`/`bytetrack`) |
| `--confidence` | `0.4`   | Detection confidence threshold |
| `--image-size` | `320`   | YOLO inference size          |
| `--max-frames` | `0`     | Stop after N frames (`0` = unlimited) |

Tracking requires the `lap` (and `filterpy`) packages, included in the `vision`
extra.

## Library

```python
from vision.tracking.client import TrackingClient
from vision.common.types import FrameResult

tracker = TrackingClient(tracker_name="yolo")
result: FrameResult = tracker.run(frame, result)   # result.detections required
print(result.tracks)                               # list[Track] with stable ids
```

## Logging

The client and tracker emit structured logs (via `shared.logging`) to
`.logs/vision_tracking.log`, including tracker initialization and per-frame
object counts.

## Extending

Add another backend by subclassing `ObjectTracker` and registering it in
`factory.py`. The Ultralytics tracker config (`.yaml`) is selected automatically
for `bytetrack`.

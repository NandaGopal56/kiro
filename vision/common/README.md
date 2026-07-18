# vision/common

Shared building blocks reused by every vision module. Nothing here touches a
camera or a model directly — it is the foundation the other modules build on.

## Files

- **`client.py`** — `VisionClient`, the common interface every module implements.
  Subclasses implement `run(frame, result=None) -> FrameResult`, which lets each
  module be used both standalone (via its `__main__`) and composed by
  `VisionPipeline`.
- **`env.py`** — `init_env()` loads `.env` with `load_dotenv(find_dotenv())` so
  the project-root path is never hardcoded and works for both CLI and app use.
- **`paths.py`** — `project_root()`, `model_dir()`, `model_path(name)`. All model
  weights resolve to the project-root `.models/` folder (created if missing).
- **`logging.py`** — re-exports `shared.logging.get_logger` so every module has a
  single logging source.
- **`types.py`** — dataclasses: `Detection`, `Track`, `Identity`, `FrameResult`.
  `FrameResult` is the unified output passed between pipeline stages.

## Usage

```python
from vision.common.client import VisionClient
from vision.common.paths import model_path
from vision.common.env import init_env

init_env()
weights = model_path("yolo11m.pt")   # -> <project root>/.models/yolo11m.pt
```

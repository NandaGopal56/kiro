# vision/common

Shared building blocks reused by every vision module. Nothing here touches a
camera or a model directly — it is the foundation the other modules build on.

## Files

- **`env.py`** — `init_env()` loads `.env` with `load_dotenv(find_dotenv())` so
  the project-root path is never hardcoded and works for both CLI and app use.
- **`paths.py`** — `project_root()`, `model_dir()`, `model_path(name)`. All model
  weights resolve to the project-root `.models/` folder (created if missing).
- **`logging.py`** — re-exports `shared.logging.get_logger` so every module has a
  single logging source.
- **`types.py`** — dataclasses: `Detection`, `Track`, `Identity`, `FrameResult`.
  `FrameResult` is the unified output passed between pipeline stages and is the
  contract used to join modules inside `VisionPipeline`.

## Usage

```python
from vision.common.paths import model_path
from vision.common.env import init_env

init_env()
weights = model_path("yolo11n.pt")   # -> <project root>/.models/yolo11n.pt
```

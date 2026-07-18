# Vision Subsystem

A modular computer-vision pipeline for detection, tracking, and vision-language
modeling (VLM). Each capability is an **independent module** with its own client
and `__main__` for standalone CLI use. The modules are joined *only* by
`VisionPipeline`, which is the single orchestrator.

## Module READMEs

Each submodule has its own README with detailed usage and extension notes:

- [`vision/common`](common/README.md) — shared env, paths, types (no behavior)
- [`vision/detection`](detection/README.md) — object detection (YOLO)
- [`vision/tracking`](tracking/README.md) — object tracking (IoU-stable IDs)
- [`vision/vlm`](vlm/README.md) — vision-language model (OpenAI via LangChain)

## Design

- **Independent modules** — `detection`, `tracking`, and `vlm` do not depend on
  each other. Each exposes a concrete client (`DetectionClient`,
  `TrackingClient`, `VLMClient`) and a `__main__` for standalone runs.
- **Joined only by the pipeline** — `vision/pipeline.py` (`VisionPipeline`) is
  the *only* place that wires the modules together, sharing a `FrameResult`
  contract (defined in `common/types.py`). There is no shared base class
  coupling the modules.
- **One orchestrator** — `VisionPipeline` runs real-time perception
  (detect → track) and calls the slower VLM on a throttled cadence.
- **Cross-subsystem join** — vision is composed with agents/STT/TTS/RAG at the
  `kiro/` runtime layer, *not* inside `vision/`. Vision keeps to perception;
  `kiro/` keeps to system orchestration.
- **Shared concerns** — `.env` loading (`vision/common/env.py`) and model paths
  (`vision/common/paths.py`) live in `common/` and are reused everywhere, with
  no hardcoded paths.

## Structure

```
vision/
├── __init__.py            # exports VisionPipeline; loads .env
├── __main__.py            # top-level CLI → VisionPipeline.run_camera
├── common/
│   ├── env.py             # init_env(): load_dotenv(find_dotenv())
│   ├── paths.py           # project_root() / model_dir() / model_path()
│   ├── logging.py         # re-export of shared.logging
│   └── types.py           # Detection, Track, Identity, FrameResult
├── detection/
│   ├── client.py          # DetectionClient (standalone entry point)
│   ├── __main__.py        # standalone detection CLI
│   ├── base.py            # ObjectDetector ABC
│   ├── factory.py         # create_detector("yolo")
│   └── yolo.py            # YOLODetector (ultralytics)
├── tracking/
│   ├── client.py          # TrackingClient (standalone entry point)
│   ├── __main__.py        # standalone detection+track CLI
│   ├── base.py            # ObjectTracker ABC
│   ├── factory.py         # create_tracker("yolo" / "bytetrack")
│   └── yolo.py            # YOLOTracker (IoU-stable IDs)
├── vlm/
│   ├── client.py          # VLMClient (standalone entry point)
│   ├── __main__.py        # standalone VLM caption CLI
│   ├── base.py            # VisionLanguageModel ABC + VLMQuery/VLMResponse
│   ├── factory.py         # create_vlm("openai")
│   └── openai.py          # OpenAIVisionLM (LangChain + gpt-4o-mini)
└── pipeline.py            # THE ONLY orchestrator (composes the clients)
```

## Data flow

```
camera frame
   │
   ▼
VisionPipeline.process(frame)
   ├── DetectionClient.run(frame)   → FrameResult.detections
   ├── TrackingClient.run(frame, r) → FrameResult.tracks
   └── (every N frames) VLMClient.run(frame, r) → FrameResult.caption
```

For module-level runs, each `__main__` calls its own client's `run` directly.

## Setup

The package is part of the `kiro` project and is installed via `uv`. A
`vision` optional-dependency group (LangChain + OpenAI) is provided.

```bash
# from the project root
uv sync --extra vision
```

This also pulls in `ultralytics` and `opencv-python` (core dependencies).

### Environment

API keys are read from `.env` at the project root. Keys are loaded with
`init_env()` (which uses `find_dotenv`, so the path is never hardcoded), so it
works whether you run a CLI module or import `vision` into an app.

```bash
# .env must contain, e.g.
OPENAI_API_KEY=sk-...
```

### Models

Model weights live in the project-root `.models/` directory (created
automatically). `model_path("yolo11m.pt")` resolves there. On first detection
run, YOLO downloads `yolo11m.pt` into `.models/` if it is missing.

## Usage

All commands are run from the project root with `uv run -m`.

### Individual modules (standalone CLI)

**Detection only:**

```bash
uv run -m vision.detection --confidence 0.4 --image-size 320 --max-frames 200
```

**Detection + tracking:**

```bash
uv run -m vision.tracking --tracker yolo --confidence 0.4 --max-frames 200
```

**VLM only (periodic scene captioning):**

```bash
uv run -m vision.vlm --backend openai --prompt "What objects are present?" --every-n 30
```

### Orchestrated pipeline (all modules together)

```bash
uv run -m vision --tracker yolo --vlm openai --caption-every-n 30
```

Useful flags for `uv run -m vision`:

| Flag                | Default   | Description                          |
|---------------------|-----------|--------------------------------------|
| `--tracker`         | `yolo`    | Tracker backend (`yolo`/`bytetrack`) |
| `--vlm`             | `openai`  | VLM backend, or `none` to disable    |
| `--confidence`      | `0.4`     | Detection confidence threshold       |
| `--image-size`      | `320`     | YOLO inference size                  |
| `--caption-every-n` | `30`      | Run VLM caption every N frames       |
| `--camera`          | `0`       | Camera device index                  |
| `--max-frames`      | `0`       | Stop after N frames (`0` = unlimited)|

Press `q` in any camera window to quit.

## Using the pipeline as a library

```python
from vision import VisionPipeline

pipeline = VisionPipeline.build(
    tracker_name="yolo",
    vlm_backend="openai",   # or None to skip VLM
    caption_every_n=30,
)

# inside a frame loop:
result = pipeline.process(frame)        # FrameResult
print(result.detections, result.tracks, result.caption)

# or a one-off VLM question:
answer = pipeline.caption(frame, "Is there a person in the frame?")
```

## Extending

- **New detector/tracker/VLM**: implement the matching ABC
  (`ObjectDetector` / `ObjectTracker` / `VisionLanguageModel`), register it in
  the factory, and add `--backend`/`--tracker` handling. The `client.py` and
  `__main__.py` for that module need no changes.
- **Real tracking**: `YOLOTracker` (`vision/tracking/yolo.py`) provides
  BoT-SORT/ByteTrack via Ultralytics' `model.track(persist=True)`. Add another
  backend by subclassing `ObjectTracker` and registering it in
  `tracking/factory.py`.
- **Recognition**: the `Identity` type already exists in
  `vision/common/types.py`; add a recognition stage to `VisionPipeline` that
  enriches `FrameResult.tracks` with identities (e.g. CLIP re-ID).
- **Agents integration**: keep `vision/` owning perception and call
  `pipeline.process` / `vlm.analyze` from an `agents/vision_agent`, matching how
  `agents` already uses STT/TTS providers.

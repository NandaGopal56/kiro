from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from shared.logging import get_logger
from vision.common.paths import model_path
from vision.common.types import FrameResult
from vision.detection.client import DetectionClient
from vision.tracking.client import TrackingClient
from vision.vlm.client import VLMClient

logger = get_logger("vision.pipeline", log_file="vision_pipeline.log")

DEFAULT_MODEL_PATH = model_path("yolo11m.pt")


class VisionPipeline:
    """Single orchestrator for the vision subsystem.

    Composes the detection, tracking and (optional) VLM *clients* — the same
    client objects each module also exposes via its own ``__main__``. There is
    therefore exactly one code path per capability, reused by both the CLI and
    this orchestrator.

    Real-time perception (detect -> track) is separated from the slower VLM
    path, which runs on a throttled cadence or on demand.
    """

    def __init__(
        self,
        detection: DetectionClient,
        tracking: TrackingClient,
        vlm: Optional[VLMClient] = None,
        caption_every_n: int = 30,
    ):
        self.detection = detection
        self.tracking = tracking
        self.vlm = vlm
        self.caption_every_n = caption_every_n
        self._frame_count = 0

    @classmethod
    def build(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        confidence: float = 0.4,
        image_size: int = 320,
        tracker_name: str = "yolo",
        vlm_backend: Optional[str] = None,
        caption_every_n: int = 30,
    ) -> "VisionPipeline":
        tracking = TrackingClient(
            tracker_name, model_path, "botsort.yaml", confidence, image_size
        )
        # Only build a separate detection model when the tracker cannot produce
        # detections in the same forward pass (avoids a redundant 2nd YOLO load).
        detection = None
        if not getattr(tracking, "_combined", False):
            detection = DetectionClient(model_path, confidence, image_size)
        vlm = VLMClient(vlm_backend) if vlm_backend else None
        return cls(detection, tracking, vlm, caption_every_n)

    def process(self, frame) -> FrameResult:
        result = FrameResult(frame=frame)
        # When the tracker cannot detect+track in one pass, run detection first.
        if self.detection is not None:
            result = self.detection.run(frame, result)
        result = self.tracking.run(frame, result)
        self._frame_count += 1
        if (
            self.vlm
            and self.caption_every_n
            and self._frame_count % self.caption_every_n == 0
        ):
            self._schedule_caption(frame, result)
        return result

    def _schedule_caption(self, frame, result: FrameResult) -> None:
        """Run the (slow) VLM in a background thread so it never stalls frames."""

        def _work() -> None:
            try:
                captioned = self.vlm.run(frame.copy(), result)
                result.caption = captioned.caption
                logger.info("Caption: %s", result.caption)
            except Exception as exc:  # keep the real-time loop alive on VLM errors
                logger.error("VLM caption failed: %s", exc)

        threading.Thread(target=_work, daemon=True).start()

    def caption(self, frame, prompt: str) -> str:
        if not self.vlm:
            raise RuntimeError("No VLM client configured.")
        return self.vlm.analyze(frame, prompt).text

    def run_camera(self, camera_index: int = 0, max_frames: int = 0) -> None:
        import cv2
        import time

        # Neutral, class-keyed palette (max 4 colors, cycled by class_id).
        COLORS = [
            (0, 0, 0),      # black
            (0, 0, 255),    # red
            (255, 0, 0),    # blue
            (0, 165, 255),  # orange
        ]

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera.")

        logger.info("Vision pipeline started")
        prev_time = time.time()
        fps = 0.0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)  # mirror: horizontal (left-right) flip
                result = self.process(frame)

                for tr in result.tracks:
                    x1, y1, x2, y2 = tr.bbox
                    color = COLORS[tr.class_id % len(COLORS)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(
                        frame, f"#{tr.track_id} {tr.class_name}",
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1,
                    )

                if result.caption:
                    cv2.putText(
                        frame, result.caption[:80], (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                    )

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

                cv2.imshow("Vision Pipeline", frame)
                if max_frames and self._frame_count >= max_frames:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Vision pipeline stopped")

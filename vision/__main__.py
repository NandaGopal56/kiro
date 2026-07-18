from __future__ import annotations

import argparse

from vision.common.env import init_env
from vision.common.logging import get_logger
from vision.pipeline import VisionPipeline

logger = get_logger("vision.__main__", log_file="vision.log")
init_env()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the vision pipeline orchestrator.")
    parser.add_argument("--tracker", default="yolo")
    parser.add_argument("--vlm", default="openai", help="VLM backend or 'none'")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--caption-every-n", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    vlm_backend = None if args.vlm.lower() == "none" else args.vlm

    pipeline = VisionPipeline.build(
        confidence=args.confidence,
        image_size=args.image_size,
        tracker_name=args.tracker,
        vlm_backend=vlm_backend,
        caption_every_n=args.caption_every_n,
    )
    logger.info("Starting vision pipeline (tracker=%s vlm=%s)", args.tracker, args.vlm)
    pipeline.run_camera(camera_index=args.camera, max_frames=args.max_frames)


if __name__ == "__main__":
    main()

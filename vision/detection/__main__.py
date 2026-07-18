from __future__ import annotations

import argparse

from dotenv import find_dotenv, load_dotenv

from shared.logging import get_logger
from vision.common.types import FrameResult
from vision.detection.client import DetectionClient

logger = get_logger("vision.detection.__main__", log_file="vision_detection.log")
load_dotenv(find_dotenv())


def _camera_loop(client: DetectionClient, max_frames: int = 0) -> None:
    import cv2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera.")

    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            result: FrameResult = client.run(frame)
            for det in result.detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{det.class_name} {det.confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )
            cv2.imshow("Detection", frame)
            count += 1
            if max_frames and count >= max_frames:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run object detection standalone.")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    client = DetectionClient(confidence=args.confidence, image_size=args.image_size)
    logger.info("Starting detection client")
    _camera_loop(client, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
